import { LANGUAGE_DETECTION_MODEL, displayLanguage, getCatalogModel } from "./catalog.js";
import { Deferred } from "./deferred.js";
import {
  detectLanguage,
  missingDetectorWarning,
  settingsLanguagePolicy,
  undetectablePreferences,
  type AutoLanguagePolicy,
} from "./language-detection-policy.js";
import type { TranscribeSettings } from "./settings.js";
import type {
  DictationStream,
  TranscribeCppBackend,
  TranscriptionOptions,
} from "./transcription.js";

/** A transcript, plus any language identification fallback or limitation. */
export type Transcript = { text: string; warning?: string };

type TranscriptionJob = {
  settings: TranscribeSettings;
  policy: AutoLanguagePolicy;
  pcm: Float32Array;
  signal?: AbortSignal;
  resolve: (transcript: Transcript) => void;
  reject: (error: unknown) => void;
  started: boolean;
  settled: boolean;
  removeAbortListener?: () => void;
};

type ReservationState = {
  settings: TranscribeSettings;
  policy: AutoLanguagePolicy;
  accepting: boolean;
  submitted: boolean;
  cancelled: boolean;
  started: boolean;
  /** Settles once the model (and stream, when available) is ready, or on failure. */
  ready: Deferred;
  /** Resolves when recording ends, whether by submit or cancel. */
  submission: Deferred;
  result: Deferred<Transcript>;
  fullPcm?: Float32Array;
  signal?: AbortSignal;
  removeAbortListener?: () => void;
  chunks: Float32Array[];
  stream?: DictationStream;
  drain?: Promise<void>;
  streamUnavailable: boolean;
  streamError?: unknown;
};

export type DictationReservation = {
  readonly ready: Promise<void>;
  feed(chunk: Float32Array): void;
  submit(pcm: Float32Array, signal?: AbortSignal): Promise<Transcript>;
  cancel(): void;
};

type ReusableBackend = Pick<
  TranscribeCppBackend,
  "prepare" | "transcribe" | "dispose"
> & {
  startStream?: (
    options?: TranscriptionOptions,
  ) => Promise<DictationStream | undefined>;
  identifyLanguage?: TranscribeCppBackend["identifyLanguage"];
};

type BackendFactory = (
  modelPath: string,
) => ReusableBackend | Promise<ReusableBackend>;

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error("Transcription cancelled");
}

function cancellationError(state: ReservationState): Error {
  return state.signal?.aborted ? abortError(state.signal) : new Error("Dictation cancelled");
}

function transcriptionOptions(
  settings: TranscribeSettings,
  language: string | undefined,
  signal?: AbortSignal,
): TranscriptionOptions {
  return { signal, language, chineseOutput: settings.chineseOutput };
}

function fixedLanguage(policy: AutoLanguagePolicy): string | undefined {
  return policy.kind === "fixed" ? policy.language : undefined;
}

async function cachedDetectorPath(): Promise<string | undefined> {
  // Keeps Hugging Face modules off the path of every other take.
  const { findCachedCatalogModel } = await import("./models.js");
  return findCachedCatalogModel(LANGUAGE_DETECTION_MODEL)?.path;
}

/** Owns the loaded model and schedules dictation ahead of queued file jobs. */
export class TranscriptionService {
  private backend: ReusableBackend | undefined;
  private modelPath: string | undefined;
  private readonly dictationQueue: ReservationState[] = [];
  private readonly fileQueue: TranscriptionJob[] = [];
  private readonly reservationStates = new Set<ReservationState>();
  private reservation: ReservationState | undefined;
  private loop: Promise<void> | undefined;
  private shuttingDown = false;
  private readonly shutdownController = new AbortController();
  /** The language detector, loaded on demand like the transcription model. */
  private detector: ReusableBackend | undefined;
  /** A missing detector warns on the first take only, not every take. */
  private missingDetectorWarned = false;

  constructor(
    private readonly createBackend: BackendFactory = async (modelPath) => {
      const { TranscribeCppBackend } = await import("./transcription.js");
      return new TranscribeCppBackend(modelPath);
    },
  ) {}

  reserveDictation(settings: TranscribeSettings): DictationReservation {
    if (this.shuttingDown) throw new Error("Pi Voice is shutting down");
    if (this.reservation) throw new Error("A dictation reservation is already active");

    const policy = settingsLanguagePolicy(settings);
    const state: ReservationState = {
      settings,
      policy,
      accepting: true,
      submitted: false,
      cancelled: false,
      started: false,
      ready: new Deferred(),
      submission: new Deferred(),
      result: new Deferred<Transcript>(),
      chunks: [],
      // Detection needs the finished clip before transcription can start.
      streamUnavailable: policy.kind === "detect",
    };
    // Cancellation can happen before callers attach handlers to either promise.
    void state.ready.promise.catch(() => undefined);
    void state.result.promise.catch(() => undefined);
    this.reservation = state;
    this.reservationStates.add(state);
    this.schedule();

    return {
      ready: state.ready.promise,
      feed: (chunk) => {
        if (
          !state.accepting ||
          state.streamUnavailable ||
          state.streamError !== undefined ||
          chunk.length === 0
        ) return;
        state.chunks.push(chunk);
        this.startDrain(state);
      },
      submit: (pcm, signal) => {
        if (!state.accepting || state.submitted) {
          return Promise.reject(new Error("The dictation reservation is no longer active"));
        }
        state.accepting = false;
        state.submitted = true;
        state.fullPcm = pcm;
        if (this.reservation === state) this.reservation = undefined;
        if (!state.started) this.dictationQueue.push(state);
        state.signal = signal;
        if (signal) {
          const onAbort = (): void => this.abortReservation(state);
          signal.addEventListener("abort", onAbort, { once: true });
          state.removeAbortListener = () => signal.removeEventListener("abort", onAbort);
          if (signal.aborted) onAbort();
        }
        state.submission.resolve();
        this.schedule();
        return state.result.promise;
      },
      cancel: () => {
        if (!state.accepting || state.submitted) return;
        state.accepting = false;
        state.cancelled = true;
        state.chunks.length = 0;
        if (this.reservation === state) this.reservation = undefined;
        this.resetStream(state);
        state.ready.reject(cancellationError(state));
        state.submission.resolve();
        if (!state.started) this.reservationStates.delete(state);
        this.schedule();
      },
    };
  }

  transcribeFile(
    settings: TranscribeSettings,
    pcm: Float32Array,
    signal?: AbortSignal,
  ): Promise<Transcript> {
    if (this.shuttingDown) return Promise.reject(new Error("Pi Voice is shutting down"));
    if (signal?.aborted) return Promise.reject(abortError(signal));

    const result = new Promise<Transcript>((resolve, reject) => {
      const job: TranscriptionJob = {
        settings,
        policy: settingsLanguagePolicy(settings),
        pcm,
        signal,
        resolve,
        reject,
        started: false,
        settled: false,
      };
      if (signal) {
        const onAbort = (): void => {
          if (job.started || job.settled) return;
          const index = this.fileQueue.indexOf(job);
          if (index >= 0) this.fileQueue.splice(index, 1);
          this.settleJob(job, () => reject(abortError(signal)));
          this.schedule();
        };
        signal.addEventListener("abort", onAbort, { once: true });
        job.removeAbortListener = () => signal.removeEventListener("abort", onAbort);
      }
      this.fileQueue.push(job);
    });
    this.schedule();
    return result;
  }

  private abortReservation(state: ReservationState): void {
    if (state.cancelled || state.result.settled) return;
    state.cancelled = true;
    state.accepting = false;
    state.chunks.length = 0;
    // Queue native reset before the scheduler can begin another model operation.
    this.resetStream(state);
    state.submission.resolve();

    // A submitted reservation waiting behind active work has no native state to
    // tear down, so preserve the old immediate-cancellation behavior.
    if (!state.started) {
      const index = this.dictationQueue.indexOf(state);
      if (index >= 0) this.dictationQueue.splice(index, 1);
      this.reservationStates.delete(state);
      state.removeAbortListener?.();
      const error = cancellationError(state);
      state.ready.reject(error);
      state.result.reject(error);
    }
  }

  private resetStream(state: ReservationState): void {
    const stream = state.stream;
    state.stream = undefined;
    stream?.reset();
  }

  private startDrain(state: ReservationState): void {
    if (
      !state.stream ||
      state.drain ||
      state.cancelled ||
      state.streamUnavailable ||
      state.streamError !== undefined ||
      state.chunks.length === 0
    ) {
      return;
    }

    const draining = this.drainStream(state).finally(() => {
      if (state.drain === draining) state.drain = undefined;
      this.startDrain(state);
    });
    state.drain = draining;
  }

  private async drainStream(state: ReservationState): Promise<void> {
    const stream = state.stream!;
    while (
      state.stream === stream &&
      !state.cancelled &&
      state.chunks.length > 0
    ) {
      const chunk = state.chunks.shift()!;
      try {
        await stream.feed(chunk);
      } catch (error) {
        state.streamError = error;
        state.chunks.length = 0;
        // A batch fallback may only start after reset has been issued.
        this.resetStream(state);
        return;
      }
    }
  }

  private settleJob(job: TranscriptionJob, settle: () => void): void {
    if (job.settled) return;
    job.settled = true;
    job.removeAbortListener?.();
    settle();
  }

  private schedule(): void {
    if (this.loop) return;
    const loop = this.process().finally(() => {
      if (this.loop === loop) this.loop = undefined;
      if (this.hasRunnableWork()) this.schedule();
    });
    this.loop = loop;
  }

  private hasRunnableWork(): boolean {
    if (this.shuttingDown) return false;
    return (
      this.dictationQueue.length > 0 ||
      this.reservation !== undefined ||
      this.fileQueue.length > 0
    );
  }

  private async process(): Promise<void> {
    // Model work is non-preemptive. At each operation boundary, submitted
    // dictation and active recording reservations run before queued file jobs.
    // An active recording holds the model lane until submit or cancel.
    while (!this.shuttingDown) {
      const queuedDictation = this.dictationQueue.shift();
      if (queuedDictation) {
        queuedDictation.started = true;
        await this.runReservation(queuedDictation);
        continue;
      }

      const reservation = this.reservation;
      if (reservation) {
        reservation.started = true;
        await this.runReservation(reservation);
        if (this.reservation === reservation) this.reservation = undefined;
        continue;
      }

      const file = this.fileQueue.shift();
      if (file) {
        await this.runJob(file);
        continue;
      }

      await this.unloadAll();
      return;
    }
  }

  private async runReservation(state: ReservationState): Promise<void> {
    try {
      await this.runReservationWork(state);
    } finally {
      state.removeAbortListener?.();
      this.reservationStates.delete(state);
    }
  }

  private settleCancelledReservation(state: ReservationState): void {
    this.resetStream(state);
    const error = cancellationError(state);
    state.ready.reject(error);
    if (state.submitted) state.result.reject(error);
  }

  private async runReservationWork(state: ReservationState): Promise<void> {
    if (state.cancelled) {
      this.settleCancelledReservation(state);
      return;
    }

    let backend: ReusableBackend;
    try {
      backend = await this.ensureModel(state.settings.model.path);
    } catch (error) {
      if (state.cancelled) {
        this.settleCancelledReservation(state);
        return;
      }
      state.streamUnavailable = true;
      state.chunks.length = 0;
      state.ready.reject(error);
      await state.submission.promise;
      if (state.submitted) state.result.reject(error);
      return;
    }

    if (state.cancelled) {
      this.settleCancelledReservation(state);
      return;
    }

    // If recording already ended while the model loaded (or while the stream
    // opened), avoid replaying the complete clip through a newly opened stream
    // and use the batch path.
    if (!state.submitted && !state.streamUnavailable && backend.startStream) {
      try {
        const stream = await backend.startStream(
          transcriptionOptions(state.settings, fixedLanguage(state.policy)),
        );
        if (stream && state.submitted) {
          stream.reset();
        } else {
          state.stream = stream;
        }
        if (!state.stream) {
          state.streamUnavailable = true;
          state.chunks.length = 0;
        }
      } catch (error) {
        // Stream setup is an optimization; preserve dictation via batch fallback.
        state.streamError = error;
        state.chunks.length = 0;
      }
    }

    if (state.cancelled) {
      this.settleCancelledReservation(state);
      return;
    }

    state.ready.resolve();
    this.startDrain(state);
    // Warm the detector while the user speaks; a failure resurfaces at detection.
    if (state.policy.kind === "detect") await this.ensureDetector().catch(() => undefined);
    await state.submission.promise;

    if (!state.submitted) {
      this.resetStream(state);
      return;
    }

    try {
      if (state.cancelled || state.signal?.aborted) throw cancellationError(state);

      this.startDrain(state);
      while (state.drain) await state.drain;

      if (state.cancelled || state.signal?.aborted) throw cancellationError(state);

      const pcm = state.fullPcm!;
      const stream = state.stream;
      // An empty clip takes the batch path so it fails exactly as it did before
      // streaming existed, instead of finalizing a stream that was never fed.
      if (stream && state.streamError === undefined && pcm.length > 0) {
        try {
          const text = await stream.finalize();
          state.signal?.throwIfAborted();
          state.result.resolve({ text });
          return;
        } catch (error) {
          if (state.signal?.aborted) throw abortError(state.signal);
          state.streamError = error;
        } finally {
          // Idempotent if finalize already released it. This must happen before
          // the reservation is released and file work is scheduled.
          if (state.stream === stream) state.stream = undefined;
          stream.reset();
        }
      }

      // Stream setup/feed/finalize failure: reset precedes the batch fallback.
      this.resetStream(state);
      state.chunks.length = 0;
      state.result.resolve(await this.transcribeBatch(
        backend,
        state.settings,
        state.policy,
        pcm,
        this.withShutdown(state.signal),
      ));
    } catch (error) {
      this.resetStream(state);
      state.result.reject(
        state.signal?.aborted ? abortError(state.signal) : error,
      );
    }
  }

  private withShutdown(signal?: AbortSignal): AbortSignal {
    return signal
      ? AbortSignal.any([signal, this.shutdownController.signal])
      : this.shutdownController.signal;
  }

  /** The single batch path shared by file jobs and the dictation fallback. */
  private async transcribeBatch(
    backend: ReusableBackend,
    settings: TranscribeSettings,
    policy: AutoLanguagePolicy,
    pcm: Float32Array,
    signal: AbortSignal,
  ): Promise<Transcript> {
    signal.throwIfAborted();
    let language = fixedLanguage(policy);
    let warning: string | undefined;
    if (policy.kind === "detect" && !this.detector && !(await cachedDetectorPath())) {
      language = policy.languages[0]!;
      if (!this.missingDetectorWarned) warning = missingDetectorWarning(language);
      this.missingDetectorWarned = true;
    } else if (policy.kind === "detect") {
      ({ language, warning } = await detectLanguage(policy, pcm, async (clip, allowed) =>
        (await this.ensureDetector()).identifyLanguage!(clip, allowed, signal)));
      signal.throwIfAborted();
    } else if (policy.kind === "unsupported") {
      // Settings normalization drops Auto first; this guards stale settings.
      const skipped = undetectablePreferences(getCatalogModel(settings.model.id)!, settings.preferredLanguages);
      const which = skipped.length ? ` (${skipped.map(displayLanguage).join(", ")})` : "";
      throw new Error(`Auto detect doesn't work with your current languages${which}. Choose a language in /voice-settings.`);
    }
    const text = await backend.transcribe(pcm, transcriptionOptions(settings, language, signal));
    return warning ? { text, warning } : { text };
  }

  private async runJob(job: TranscriptionJob): Promise<void> {
    if (job.settled) return;
    job.started = true;
    const signal = this.withShutdown(job.signal);

    try {
      signal.throwIfAborted();
      const backend = await this.ensureModel(job.settings.model.path);
      const transcript = await this.transcribeBatch(backend, job.settings, job.policy, job.pcm, signal);
      this.settleJob(job, () => job.resolve(transcript));
    } catch (error) {
      this.settleJob(job, () => job.reject(error));
    }
  }

  /** Called only from the scheduler loop, so loads never race. */
  private async ensureDetector(): Promise<ReusableBackend> {
    if (!this.detector) {
      const path = await cachedDetectorPath();
      if (!path) throw new Error("the language detector isn't downloaded");
      this.detector = await this.createBackend(path);
    }
    await this.detector.prepare();
    return this.detector;
  }

  private async ensureModel(modelPath: string): Promise<ReusableBackend> {
    if (this.backend && this.modelPath === modelPath) {
      await this.backend.prepare();
      return this.backend;
    }

    await this.unloadModel();
    const backend = await this.createBackend(modelPath);
    this.backend = backend;
    this.modelPath = modelPath;
    try {
      await backend.prepare();
      return backend;
    } catch (error) {
      if (this.backend === backend) {
        this.backend = undefined;
        this.modelPath = undefined;
      }
      await backend.dispose().catch(() => undefined);
      throw error;
    }
  }

  private async unloadModel(): Promise<void> {
    const backend = this.backend;
    if (!backend) return;
    this.backend = undefined;
    this.modelPath = undefined;
    await backend.dispose();
  }

  /** Never rejects: a dispose failure would otherwise go unobserved. */
  private async unloadAll(): Promise<void> {
    const detector = this.detector;
    this.detector = undefined;
    await Promise.all([
      this.unloadModel().catch(() => undefined),
      detector?.dispose().catch(() => undefined),
    ]);
  }

  async shutdown(): Promise<void> {
    if (!this.shuttingDown) {
      this.shuttingDown = true;
      this.shutdownController.abort(new Error("Pi Voice is shutting down"));
      const shutdownError = new Error("Pi Voice is shutting down");
      this.reservation = undefined;
      this.dictationQueue.length = 0;
      for (const reservation of this.reservationStates) {
        reservation.accepting = false;
        reservation.cancelled = true;
        reservation.chunks.length = 0;
        this.resetStream(reservation);
        reservation.ready.reject(shutdownError);
        reservation.result.reject(shutdownError);
        reservation.submission.resolve();
      }
      for (const job of this.fileQueue) {
        this.settleJob(job, () => job.reject(shutdownError));
      }
      this.fileQueue.length = 0;
    }

    await this.loop?.catch(() => undefined);
    this.reservationStates.clear();
    await this.unloadAll();
  }
}
