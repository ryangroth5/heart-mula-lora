const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');

const app = express();
const PORT = 3000;
const BASE_DIR = __dirname;

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(BASE_DIR, 'public')));

// In-memory job registry
const jobs = {};
let activeJob = null;
let activeTrainJob = null;

// POST /generate — start a generation job
app.post('/generate', (req, res) => {
  // Kill any running job to free GPU memory before starting a new one
  if (activeJob && activeJob.proc) {
    try { activeJob.proc.kill('SIGKILL'); } catch {}
    activeJob = null;
  }

  const p = req.body;
  const jobId = crypto.randomBytes(8).toString('hex');

  // Write lyrics and tags to temp files
  const tmpDir = os.tmpdir();
  const lyricsFile = path.join(tmpDir, `heartmula-lyrics-${jobId}.txt`);
  const tagsFile = path.join(tmpDir, `heartmula-tags-${jobId}.txt`);
  fs.writeFileSync(lyricsFile, p.lyrics || '');
  fs.writeFileSync(tagsFile, p.tags || '');

  const args = [
    path.join(BASE_DIR, 'examples', 'run_music_generation.py'),
    `--model_path=${p.model_path || './ckpt'}`,
    `--version=${p.version || '3B'}`,
    `--lyrics=${lyricsFile}`,
    `--tags=${tagsFile}`,
    `--save_path=${p.save_path || './assets/output.mp3'}`,
    `--max_audio_length_ms=${p.max_audio_length_ms || 240000}`,
    `--temperature=${p.temperature || 1.0}`,
    `--topk=${p.topk || 50}`,
    `--cfg_scale=${p.cfg_scale || 1.5}`,
    `--seed=${p.seed !== undefined ? p.seed : -1}`,
    `--mula_device=${p.mula_device || 'cuda'}`,
    `--codec_device=${p.codec_device || 'cuda'}`,
    `--mula_dtype=${p.mula_dtype || 'bf16'}`,
    `--codec_dtype=${p.codec_dtype || 'fp32'}`,
  ];
  if (p.lazy_load) args.push('--lazy_load');
  if (p.lora_path) {
    args.push(`--lora_path=${p.lora_path}`);
    args.push(`--lora_rank=${p.lora_rank || 8}`);
    args.push(`--lora_alpha=${p.lora_alpha || 16}`);
  }

  const python = path.join(BASE_DIR, 'venv', 'bin', 'python');
  const proc = spawn(python, args, {
    cwd: BASE_DIR,
    env: { ...process.env, PYTORCH_ALLOC_CONF: 'expandable_segments:True', CUDA_LAUNCH_BLOCKING: '1' },
  });

  const logLines = [];
  let exitCode = null;

  const appendLine = (line) => {
    logLines.push(line);
    if (jobs[jobId]) jobs[jobId].notify(line);
  };

  proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(appendLine));
  proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(l => appendLine(`[stderr] ${l}`)));

  proc.on('close', (code) => {
    exitCode = code;
    appendLine(`__EXIT__:${code}`);
    // Clean up temp files
    try { fs.unlinkSync(lyricsFile); } catch {}
    try { fs.unlinkSync(tagsFile); } catch {}
  });

  jobs[jobId] = {
    proc,
    logLines,
    exitCode,
    savePath: p.save_path || './assets/output.mp3',
    clients: [],
    notify(line) {
      this.clients.forEach(send => send(line));
    }
  };
  activeJob = jobs[jobId];

  res.json({ jobId });
});

// GET /stream/:jobId — SSE log stream
app.get('/stream/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).send('Job not found');

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (line) => res.write(`data: ${JSON.stringify(line)}\n\n`);

  // Replay buffered lines
  job.logLines.forEach(send);

  // If already done, close immediately
  if (job.exitCode !== null) {
    res.end();
    return;
  }

  job.clients.push(send);

  req.on('close', () => {
    job.clients = job.clients.filter(c => c !== send);
  });
});

// POST /encode — encode dataset manifest → token .pt files
app.post('/encode', (req, res) => {
  const p = req.body;
  const jobId = crypto.randomBytes(8).toString('hex');

  const args = [
    path.join(BASE_DIR, 'examples', 'encode_dataset.py'),
    `--manifest=${p.manifest || './data/dataset.json'}`,
    `--output_dir=${p.output_dir || './data/tokens'}`,
    `--model_path=${p.model_path || './ckpt'}`,
  ];

  const python = path.join(BASE_DIR, 'venv', 'bin', 'python');
  const proc = spawn(python, args, {
    cwd: BASE_DIR,
    env: { ...process.env, PYTORCH_ALLOC_CONF: 'expandable_segments:True' },
  });

  const logLines = [];
  let exitCode = null;

  const appendLine = (line) => {
    logLines.push(line);
    if (jobs[jobId]) jobs[jobId].notify(line);
  };

  proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(appendLine));
  proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(l => appendLine(`[stderr] ${l}`)));

  proc.on('close', (code) => {
    exitCode = code;
    appendLine(`__EXIT__:${code}`);
  });

  jobs[jobId] = {
    proc, logLines, exitCode,
    clients: [],
    notify(line) { this.clients.forEach(send => send(line)); }
  };

  res.json({ jobId });
});

// GET /encode-stream/:jobId
app.get('/encode-stream/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).send('Job not found');

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (line) => res.write(`data: ${JSON.stringify(line)}\n\n`);
  job.logLines.forEach(send);
  if (job.exitCode !== null) { res.end(); return; }
  job.clients.push(send);
  req.on('close', () => { job.clients = job.clients.filter(c => c !== send); });
});

// POST /train — start a LoRA training job
app.post('/train', (req, res) => {
  // Kill any running training job before starting a new one
  if (activeTrainJob && activeTrainJob.proc) {
    try { activeTrainJob.proc.kill('SIGKILL'); } catch {}
    activeTrainJob = null;
  }

  const p = req.body;
  const jobId = crypto.randomBytes(8).toString('hex');

  const args = [
    path.join(BASE_DIR, 'examples', 'train_lora.py'),
    `--model_path=${p.model_path || './ckpt'}`,
    `--version=${p.version || '3B'}`,
    `--dataset_dir=${p.dataset_dir || './data/tokens'}`,
    `--output=${p.output || './lora.pt'}`,
    `--epochs=${p.epochs || 3}`,
    `--lora_rank=${p.lora_rank || 8}`,
    `--lora_alpha=${p.lora_alpha || 16}`,
    `--lr=${p.lr || 1e-4}`,
    `--grad_accum=${p.grad_accum || 4}`,
    `--ckpt_every=${p.ckpt_every !== undefined ? p.ckpt_every : 100}`,
  ];

  const python = path.join(BASE_DIR, 'venv', 'bin', 'python');
  const proc = spawn(python, args, {
    cwd: BASE_DIR,
    env: { ...process.env, PYTORCH_ALLOC_CONF: 'expandable_segments:True' },
  });

  const logLines = [];
  let exitCode = null;

  const appendLine = (line) => {
    logLines.push(line);
    if (jobs[jobId]) jobs[jobId].notify(line);
  };

  proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(appendLine));
  proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(l => appendLine(`[stderr] ${l}`)));

  proc.on('close', (code) => {
    exitCode = code;
    appendLine(`__EXIT__:${code}`);
  });

  jobs[jobId] = {
    proc,
    logLines,
    exitCode,
    clients: [],
    notify(line) {
      this.clients.forEach(send => send(line));
    }
  };
  activeTrainJob = jobs[jobId];

  res.json({ jobId });
});

// GET /train-stream/:jobId — SSE log stream for training jobs
app.get('/train-stream/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).send('Job not found');

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (line) => res.write(`data: ${JSON.stringify(line)}\n\n`);

  // Replay buffered lines
  job.logLines.forEach(send);

  // If already done, close immediately
  if (job.exitCode !== null) {
    res.end();
    return;
  }

  job.clients.push(send);

  req.on('close', () => {
    job.clients = job.clients.filter(c => c !== send);
  });
});

// POST /download — download YouTube URLs via yt-dlp into a target directory
app.post('/download', (req, res) => {
  const p = req.body;
  const urls = (p.urls || '').trim().split(/\s+/).filter(Boolean);
  if (!urls.length) return res.status(400).json({ error: 'No URLs provided' });

  const outputDir = p.output_dir || './data/raw';
  const jobId = crypto.randomBytes(8).toString('hex');
  const logLines = [];
  let exitCode = null;

  const appendLine = (line) => {
    logLines.push(line);
    if (jobs[jobId]) jobs[jobId].notify(line);
  };

  jobs[jobId] = {
    proc: null, logLines, exitCode,
    clients: [],
    notify(line) { this.clients.forEach(send => send(line)); }
  };

  res.json({ jobId });

  // Run yt-dlp sequentially for each URL via a shell command chain
  const args = [
    '-f', '140/bestaudio[ext=m4a]/bestaudio/18',
    '-x',
    '--audio-format', 'm4a',
    '--no-playlist',
    '--extractor-args', 'youtube:player_client=web',
    '--no-js-runtimes', '--js-runtimes', 'node',
    '--remote-components', 'ejs:github',
    '-o', path.join(outputDir, '%(title).50s.%(ext)s'),
  ].concat(urls);

  const ytdlp = path.join(BASE_DIR, 'venv', 'bin', 'yt-dlp');
  const proc = spawn(ytdlp, args, { cwd: BASE_DIR });
  jobs[jobId].proc = proc;

  proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(appendLine));
  proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(l => appendLine(`[stderr] ${l}`)));
  proc.on('close', (code) => {
    exitCode = code;
    jobs[jobId].exitCode = code;
    appendLine(`__EXIT__:${code}`);
  });
});

// GET /download-stream/:jobId
app.get('/download-stream/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).send('Job not found');
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();
  const send = (line) => res.write(`data: ${JSON.stringify(line)}\n\n`);
  job.logLines.forEach(send);
  if (job.exitCode !== null) { res.end(); return; }
  job.clients.push(send);
  req.on('close', () => { job.clients = job.clients.filter(c => c !== send); });
});

// GET /dataset-files — list audio files in data/raw
app.get('/dataset-files', (req, res) => {
  const dir = path.join(BASE_DIR, 'data', 'raw');
  if (!fs.existsSync(dir)) return res.json([]);
  const files = fs.readdirSync(dir)
    .filter(f => /\.(mp3|wav|flac|ogg|m4a)$/i.test(f))
    .sort()
    .map(f => ({ name: f, path: `./data/raw/${f}` }));
  res.json(files);
});

// GET /dataset-meta — read sidecar JSON for a file, falling back to dataset.json
app.get('/dataset-meta', (req, res) => {
  const file = req.query.file;
  if (!file) return res.status(400).json({});
  const metaPath = path.join(BASE_DIR, 'data', 'raw', file.replace(/\.[^.]+$/, '.json'));
  if (fs.existsSync(metaPath)) return res.json(JSON.parse(fs.readFileSync(metaPath, 'utf8')));
  // Fall back to dataset.json
  const manifestPath = path.join(BASE_DIR, 'data', 'dataset.json');
  if (fs.existsSync(manifestPath)) {
    const entries = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const entry = entries.find(e => path.basename(e.audio) === file);
    if (entry) return res.json({ tags: entry.tags || '', lyrics: entry.lyrics || '' });
  }
  res.json({ tags: '', lyrics: '' });
});

// POST /dataset-meta — save sidecar JSON for a file
app.post('/dataset-meta', (req, res) => {
  const { file, tags, lyrics } = req.body;
  if (!file) return res.status(400).json({ error: 'file required' });
  const metaPath = path.join(BASE_DIR, 'data', 'raw', file.replace(/\.[^.]+$/, '.json'));
  fs.writeFileSync(metaPath, JSON.stringify({ tags: tags || '', lyrics: lyrics || '' }, null, 2));
  res.json({ ok: true });
});

// POST /prepare — run prepare_lora_dataset.py pipeline
app.post('/prepare', (req, res) => {
  const p = req.body;
  const jobId = crypto.randomBytes(8).toString('hex');
  const logLines = [];
  let exitCode = null;

  const appendLine = (line) => {
    logLines.push(line);
    if (jobs[jobId]) jobs[jobId].notify(line);
  };

  jobs[jobId] = { proc: null, logLines, exitCode, clients: [], notify(line) { this.clients.forEach(s => s(line)); } };
  res.json({ jobId });

  const args = [
    path.join(BASE_DIR, 'examples', 'prepare_lora_dataset.py'),
    `--input_dir=${p.input_dir || './data/raw'}`,
    `--output_dir=${p.output_dir || './data/processed'}`,
    `--output_json=${p.output_json || './data/dataset.json'}`,
    `--model_path=${p.model_path || './ckpt'}`,
    `--device=${p.device || 'cuda'}`,
    `--dtype=${p.dtype || 'fp32'}`,
  ];
  if (p.skip_existing) args.push('--skip_existing');
  if (p.skip_tags)     args.push('--skip_tags');
  if (p.skip_lyrics)   args.push('--skip_lyrics');

  const python = path.join(BASE_DIR, 'venv', 'bin', 'python');
  const proc = spawn(python, args, { cwd: BASE_DIR, env: { ...process.env, PYTORCH_ALLOC_CONF: 'expandable_segments:True' } });
  jobs[jobId].proc = proc;
  proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(appendLine));
  proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(l => appendLine(`[stderr] ${l}`)));
  proc.on('close', (code) => { jobs[jobId].exitCode = code; appendLine(`__EXIT__:${code}`); });
});

// GET /prepare-stream/:jobId
app.get('/prepare-stream/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).send('Job not found');
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();
  const send = (line) => res.write(`data: ${JSON.stringify(line)}\n\n`);
  job.logLines.forEach(send);
  if (job.exitCode !== null) { res.end(); return; }
  job.clients.push(send);
  req.on('close', () => { job.clients = job.clients.filter(c => c !== send); });
});

// POST /prepend-tags — prepend tags to all entries in dataset.json
app.post('/prepend-tags', (req, res) => {
  const { tags, manifest } = req.body;
  if (!tags || !tags.trim()) return res.status(400).json({ error: 'tags required' });
  const manifestPath = path.resolve(BASE_DIR, manifest || './data/dataset.json');
  if (!fs.existsSync(manifestPath)) return res.status(404).json({ error: 'manifest not found' });

  const newTags = tags.split(',').map(t => t.trim()).filter(Boolean);
  const entries = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  let updated = 0;
  for (const entry of entries) {
    const existing = (entry.tags || '').split(',').map(t => t.trim()).filter(Boolean);
    const existingLower = new Set(existing.map(t => t.toLowerCase()));
    const toPrepend = newTags.filter(t => !existingLower.has(t.toLowerCase()));
    if (toPrepend.length) {
      entry.tags = [...toPrepend, ...existing].join(', ');
      updated++;
    }
  }
  fs.writeFileSync(manifestPath, JSON.stringify(entries, null, 2));
  res.json({ ok: true, total: entries.length, updated });
});

// POST /dataset-manifest — build dataset.json from all annotated files
app.post('/dataset-manifest', (req, res) => {
  const rawDir = path.join(BASE_DIR, 'data', 'raw');
  const outPath = path.join(BASE_DIR, 'data', 'dataset.json');
  if (!fs.existsSync(rawDir)) return res.status(400).json({ error: 'data/raw not found' });
  const entries = fs.readdirSync(rawDir)
    .filter(f => /\.(mp3|wav|flac|ogg|m4a)$/i.test(f))
    .sort()
    .map(f => {
      const metaPath = path.join(rawDir, f.replace(/\.[^.]+$/, '.json'));
      const meta = fs.existsSync(metaPath) ? JSON.parse(fs.readFileSync(metaPath, 'utf8')) : {};
      return { audio: `./data/raw/${f}`, lyrics: meta.lyrics || '', tags: meta.tags || '' };
    });
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(entries, null, 2));
  res.json({ ok: true, count: entries.length, path: './data/dataset.json' });
});

// GET /audio — serve any audio file under BASE_DIR
app.get('/audio', (req, res) => {
  const filePath = path.resolve(BASE_DIR, req.query.path || './assets/output.mp3');
  if (!filePath.startsWith(BASE_DIR)) return res.status(403).send('Forbidden');
  if (!fs.existsSync(filePath)) return res.status(404).send('Audio not found');
  res.sendFile(filePath);
});

// GET /library — list generated audio files in assets/
app.get('/library', (req, res) => {
  const assetsDir = path.join(BASE_DIR, 'assets');
  let files = [];
  if (fs.existsSync(assetsDir)) {
    files = fs.readdirSync(assetsDir)
      .filter(f => /\.(mp3|wav|flac|ogg)$/i.test(f))
      .map(f => {
        const stat = fs.statSync(path.join(assetsDir, f));
        return { name: f, path: `./assets/${f}`, mtime: stat.mtimeMs };
      })
      .sort((a, b) => b.mtime - a.mtime);
  }
  res.json(files);
});

app.listen(PORT, () => {
  console.log(`HeartMuLa UI running at http://localhost:${PORT}`);
});
