// Client webcam: gửi JPEG lên /ws/pose (stop-and-wait), nhận khớp + lệnh điều khiển.
const SKELETON = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
const SEND_W = 480, SEND_H = 360, MIN_SCORE = 0.3;

export class PoseClient {
  constructor(video, overlay, statsEl) {
    this.video = video;
    this.overlay = overlay;
    this.ctx = overlay.getContext('2d');
    this.statsEl = statsEl;
    this.grab = document.createElement('canvas');
    this.grab.width = SEND_W; this.grab.height = SEND_H;
    this.gctx = this.grab.getContext('2d');
    this.latest = null;       // kết quả mới nhất từ server
    this.running = false;
    this.fps = 0; this._frames = 0; this._t0 = performance.now();
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
    this.video.srcObject = stream;
    await this.video.play();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/pose`);
    this.ws.binaryType = 'arraybuffer';
    await new Promise((ok, fail) => { this.ws.onopen = ok; this.ws.onerror = fail; });
    this.ws.onmessage = (e) => this._onResult(JSON.parse(e.data));
    this.ws.onclose = () => { this.running = false; };
    this.running = true;
    this._send();
  }

  _send() {
    if (!this.running || this.ws.readyState !== 1) return;
    this.gctx.drawImage(this.video, 0, 0, SEND_W, SEND_H);
    this._sentAt = performance.now();
    this.grab.toBlob((b) => b && b.arrayBuffer().then((buf) => this.ws.send(buf)), 'image/jpeg', 0.8);
  }

  _onResult(r) {
    r.rtt = performance.now() - this._sentAt;
    this.latest = r;
    this._frames++;
    const now = performance.now();
    if (now - this._t0 > 500) {
      this.fps = this._frames * 1000 / (now - this._t0);
      this._frames = 0; this._t0 = now;
      this.statsEl.textContent = `pose ${this.fps.toFixed(0)} FPS · model ${r.latency_ms} ms · RTT ${r.rtt.toFixed(0)} ms`;
    }
    this._draw(r);
    this._send(); // gửi khung kế tiếp ngay khi nhận kết quả
  }

  // Vẽ skeleton lên canvas phủ trên video (cùng object-fit: cover + lật gương bằng CSS)
  _draw(r) {
    const c = this.overlay, ctx = this.ctx;
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    ctx.clearRect(0, 0, W, H);
    const vw = this.video.videoWidth || 640, vh = this.video.videoHeight || 480;
    const s = Math.max(W / vw, H / vh), ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
    const P = r.keypoints.map(([x, y, sc]) => [ox + x * vw * s, oy + y * vh * s, sc]);
    ctx.lineWidth = 3; ctx.strokeStyle = '#5fe0e8'; ctx.lineCap = 'round';
    for (const [a, b] of SKELETON) {
      if (P[a][2] < MIN_SCORE || P[b][2] < MIN_SCORE || a < 5) continue; // giống video: chỉ thân người
      ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke();
    }
    ctx.fillStyle = '#f6c33b';
    for (let j = 5; j < 17; j++) {
      if (P[j][2] < MIN_SCORE) continue;
      ctx.beginPath(); ctx.arc(P[j][0], P[j][1], 5, 0, Math.PI * 2); ctx.fill();
    }
  }

  get control() { return this.latest ? this.latest.control : null; }
}
