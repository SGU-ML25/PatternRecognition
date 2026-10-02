import * as THREE from 'three';
import { Bird } from './bird.js';
import { PoseClient } from './pose.js';
import { GATE_RADIUS, WORLD_SIZE, buildCourse, buildWorld, styleGates, terrainHeight } from './world.js';

const $ = (id) => document.getElementById(id);
const GATES = 16;

// ---------- Three.js ----------
const canvas = $('game');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, 1, 0.5, 4200);
const { skyGroup } = buildWorld(scene);
let gates = buildCourse(scene, GATES);
const bird = new Bird();
scene.add(bird.root);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

// Vệt gió tạo cảm giác tốc độ
const windN = 140, windPos = new Float32Array(windN * 6);
const windGeo = new THREE.BufferGeometry();
windGeo.setAttribute('position', new THREE.BufferAttribute(windPos, 3));
const wind = new THREE.LineSegments(windGeo, new THREE.LineBasicMaterial({ color: '#cfe0ff', transparent: true, opacity: 0.35 }));
scene.add(wind);
const windSeeds = Array.from({ length: windN }, () => new THREE.Vector3((Math.random() - 0.5) * 60, (Math.random() - 0.5) * 30, -Math.random() * 120));

// ---------- Trạng thái ----------
const state = {
  mode: 'menu', input: 'keyboard',
  pos: new THREE.Vector3(0, 0, 60), yaw: 0, pitch: 0, roll: 0, vy: 0, speed: 26,
  bank: 0, flap: 0, dive: 0, active: 0, startT: 0, elapsed: 0, lostT: 0, calib: 0,
};
function resetFlight() {
  state.pos.set(0, terrainHeight(0, 60) + 35, 60);
  state.yaw = 0; state.pitch = 0; state.roll = 0; state.vy = 0; state.speed = 26;
  state.active = 0; state.elapsed = 0; state.snapCam = true;
  for (const g of gates) g.passed = false;
}
resetFlight();

// ---------- Bàn phím ----------
const keys = new Set();
addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'Space') { state.kbFlap = 1; e.preventDefault(); }
});
addEventListener('keyup', (e) => keys.delete(e.code));

function keyboardControl(dt) {
  const target = (keys.has('ArrowLeft') || keys.has('KeyA') ? 1 : 0) - (keys.has('ArrowRight') || keys.has('KeyD') ? 1 : 0);
  state.kbBank = (state.kbBank || 0) + (target - (state.kbBank || 0)) * (1 - Math.exp(-6 * dt));
  state.kbFlap = Math.max(0, (state.kbFlap || 0) - dt * 2.2);
  const dive = keys.has('ArrowDown') || keys.has('KeyS') ? 1 : 0;
  return { present: true, bank: state.kbBank, flap: state.kbFlap, dive, wing_left: null, wing_right: null, keyboard: true };
}

// ---------- Pose (webcam -> API) ----------
const pose = new PoseClient($('cam'), $('cam-overlay'), $('pose-stats'));

// ---------- Vật lý bay ----------
const FWD = new THREE.Vector3();
function fly(dt, c, autopilot) {
  if (autopilot) c = { present: true, bank: 0, flap: 0, dive: 0, wing_left: null, wing_right: null };
  const k = 1 - Math.exp(-5 * dt);
  state.bank += (c.bank - state.bank) * k;
  state.dive += (c.dive - state.dive) * k;
  state.flap = c.flap;

  state.roll += (state.bank * 0.95 - state.roll) * (1 - Math.exp(-4 * dt));
  state.yaw += Math.sin(state.roll) * 1.25 * dt;            // nghiêng cánh -> rẽ

  // Vận tốc thẳng đứng: lượn chìm chậm, vỗ cánh đẩy lên, bổ nhào lao xuống
  let vyTarget = autopilot ? (state.pos.y < terrainHeight(state.pos.x, state.pos.z) + 30 ? 3 : 0) : -2.2;
  vyTarget -= state.dive * 22;
  state.vy += (vyTarget - state.vy) * (1 - Math.exp(-1.2 * dt));
  state.vy += state.flap * 30 * dt;
  state.vy = THREE.MathUtils.clamp(state.vy, -26, 14);

  // Tốc độ: bổ nhào tăng tốc, vỗ cánh/leo cao làm chậm dần về tốc độ chuẩn
  const targetSpeed = 26 + state.dive * 22 - Math.max(0, state.vy) * 0.4;
  state.speed += (targetSpeed - state.speed) * (1 - Math.exp(-0.8 * dt));

  FWD.set(-Math.sin(state.yaw), 0, -Math.cos(state.yaw));
  state.pos.addScaledVector(FWD, state.speed * dt);
  state.pos.y += state.vy * dt;

  const ground = terrainHeight(state.pos.x, state.pos.z) + 3;
  if (state.pos.y < ground) { state.pos.y = ground; state.vy = Math.max(state.vy, 2); state.speed *= 0.97; }
  state.pos.y = Math.min(state.pos.y, 320);
  // giữ trong bản đồ: quay đầu nhẹ khi ra rìa
  const lim = WORLD_SIZE * 0.45;
  if (Math.abs(state.pos.x) > lim || Math.abs(state.pos.z) > lim) {
    const toC = Math.atan2(state.pos.x, state.pos.z);
    state.yaw += Math.atan2(Math.sin(toC - state.yaw), Math.cos(toC - state.yaw)) * dt;
  }
  state.pitch = THREE.MathUtils.clamp(Math.atan2(state.vy, state.speed), -0.9, 0.6);

  bird.root.position.copy(state.pos);
  bird.root.rotation.y = state.yaw;
  bird.body.rotation.order = 'YXZ';
  bird.body.rotation.x = state.pitch;
  bird.body.rotation.z = state.roll;
  bird.animate(dt, { flap: state.flap, dive: state.dive,
    wingL: c.wing_left ?? null, wingR: c.wing_right ?? null });
}

// ---------- Camera ----------
const camOffset = new THREE.Vector3(), lookAt = new THREE.Vector3();
function updateCamera(dt) {
  camOffset.set(0, 3.2 + state.pitch * -2, 12).applyAxisAngle(THREE.Object3D.DEFAULT_UP, state.yaw);
  const target = state.pos.clone().add(camOffset);
  const gy = terrainHeight(target.x, target.z) + 2;
  target.y = Math.max(target.y, gy);
  if (state.snapCam) { camera.position.copy(target); state.snapCam = false; }
  camera.position.lerp(target, 1 - Math.exp(-5 * dt));
  lookAt.copy(state.pos).addScaledVector(FWD, 10); lookAt.y += 1.2;
  camera.lookAt(lookAt);
  camera.rotateZ(state.roll * 0.25);
  camera.fov += (70 + (state.speed - 26) * 0.6 - camera.fov) * (1 - Math.exp(-3 * dt));
  camera.updateProjectionMatrix();
  skyGroup.position.copy(camera.position);
}

function updateWind(dt) {
  const q = camera.quaternion;
  for (let i = 0; i < windN; i++) {
    const s = windSeeds[i];
    s.z += state.speed * dt * 2.2;
    if (s.z > 6) s.set((Math.random() - 0.5) * 60, (Math.random() - 0.5) * 30, -120);
    const a = s.clone().applyQuaternion(q).add(camera.position);
    const b = s.clone().add(new THREE.Vector3(0, 0, -2 - state.speed * 0.12)).applyQuaternion(q).add(camera.position);
    windPos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
  }
  windGeo.attributes.position.needsUpdate = true;
  wind.material.opacity = THREE.MathUtils.clamp((state.speed - 20) / 40, 0.1, 0.55);
}

// ---------- Cổng + HUD ----------
let toastTimer = 0;
function toast(msg, sec = 1.3) { $('toast').textContent = msg; $('toast').classList.add('show'); toastTimer = sec; }

function checkGates() {
  const g = gates[state.active];
  if (!g) return;
  if (state.pos.distanceTo(g.pos) < GATE_RADIUS + 3) {
    g.passed = true;
    state.active++;
    toast(state.active >= GATES ? 'Về đích!' : `Gate ${state.active}!`);
    if (state.active >= GATES) finish();
  }
}

const markerEls = [0, 1].map((i) => {
  const el = document.createElement('div');
  el.className = 'marker' + (i ? ' next' : '');
  el.innerHTML = '<div class="ring"></div><div class="arrow"></div><span></span>';
  $('markers').appendChild(el);
  return el;
});
const tmp = new THREE.Vector3();
function updateMarkers() {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  markerEls.forEach((el, i) => {
    const g = gates[state.active + i];
    if (!g || state.mode === 'menu') { el.style.display = 'none'; return; }
    el.style.display = '';
    tmp.copy(g.pos).project(camera);
    const behind = tmp.z > 1;
    let x = tmp.x, y = tmp.y;
    if (behind) { x = -x; y = -y; }
    const onScreen = !behind && Math.abs(x) < 0.92 && Math.abs(y) < 0.88;
    if (!onScreen) {        // đặt mũi tên ở mép màn hình, chỉ về phía cổng
      const m = Math.max(Math.abs(x) / 0.88, Math.abs(y) / 0.8, 1e-3);
      x /= m; y /= m;
    }
    el.style.left = `${(x * 0.5 + 0.5) * W}px`;
    el.style.top = `${(-y * 0.5 + 0.5) * H}px`;
    el.querySelector('.ring').style.display = onScreen ? '' : 'none';
    const arrow = el.querySelector('.arrow');
    arrow.style.display = onScreen ? 'none' : '';
    arrow.style.transform = `rotate(${Math.atan2(x, y)}rad)`;
    el.querySelector('span').textContent = `${Math.round(state.pos.distanceTo(g.pos))} m`;
  });
}

function fmt(t) { return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`; }

function updateHud(c) {
  $('gate-count').textContent = `${Math.min(state.active, GATES)}/${GATES}`;
  $('timer').textContent = fmt(state.elapsed);
  const agl = state.pos.y - terrainHeight(state.pos.x, state.pos.z);
  $('hud-alt').textContent = `độ cao ${agl.toFixed(0)} m · ${(state.speed * 3.6).toFixed(0)} km/h`;
  $('m-bank-l').value = Math.max(0, c.bank); $('m-bank-r').value = Math.max(0, -c.bank);
  $('m-flap').value = c.flap; $('m-dive').value = c.dive;
}

// ---------- Luồng màn hình ----------
function show(id, on) { $(id).classList.toggle('hidden', !on); }

function startPlay() {
  resetFlight();
  state.mode = 'countdown'; state.count = 3.2;
  show('menu', false); show('calib', false); show('end', false);
}
function finish() {
  state.mode = 'end';
  $('end-msg').textContent = `Qua ${GATES} cổng trong ${fmt(state.elapsed)} (${state.elapsed.toFixed(1)} giây)`;
  setTimeout(() => show('end', true), 900);
}

$('btn-kb').onclick = () => { state.input = 'keyboard'; startPlay(); };
$('btn-cam').onclick = async () => {
  $('btn-cam').disabled = true;
  try {
    await pose.start();
    $('cam-off').style.display = 'none';
    state.input = 'webcam';
    state.mode = 'calib'; state.calib = 0;
    show('menu', false); show('calib', true);
  } catch (e) {
    $('server-status').textContent = 'Không mở được webcam / server: ' + e;
    $('server-status').className = 'status err';
    $('btn-cam').disabled = false;
  }
};
$('btn-again').onclick = () => {
  if (state.input === 'webcam') { state.mode = 'calib'; state.calib = 0; show('end', false); show('calib', true); resetFlight(); }
  else startPlay();
};

fetch('/api/model/info').then((r) => r.json()).then((m) => {
  const ap = m.coco_val ? ` · COCO AP ${(m.coco_val.AP * 100).toFixed(1)}` : '';
  $('server-status').textContent = `Model: ${m.backbone} (epoch ${m.epoch}${ap}) · ${m.device}`;
  $('server-status').className = 'status ok';
}).catch(() => {
  $('server-status').textContent = 'Không kết nối được API model — chỉ chơi được bằng bàn phím';
  $('server-status').className = 'status err';
  $('btn-cam').disabled = true;
});

// ---------- Vòng lặp chính ----------
const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  let c = state.input === 'webcam' ? pose.control : keyboardControl(dt);
  const present = !!(c && c.present);
  if (!c) c = { present: false, bank: 0, flap: 0, dive: 0, wing_left: null, wing_right: null };
  if (state.input === 'webcam' && !present) { c.wing_left = null; c.wing_right = null; }

  if (state.mode === 'calib') {
    const ok = present && c.spread > 3.0;
    state.calib = THREE.MathUtils.clamp(state.calib + (ok ? dt : -dt) / 1.2, 0, 1);
    $('calib-bar').style.width = `${state.calib * 100}%`;
    $('calib-title').textContent = !present ? 'Đứng lùi lại cho thấy nửa người trên' : ok ? 'Giữ nguyên…' : 'Dang ngang 2 tay';
    if (state.calib >= 1) startPlay();
  }
  if (state.mode === 'countdown') {
    const n = Math.ceil(state.count - 0.2);
    toast(n > 0 ? String(n) : 'Bay!', 0.3);
    state.count -= dt;
    if (state.count <= 0) { state.mode = 'play'; state.startT = t; }
  }

  const playing = state.mode === 'play';
  if (playing) {
    state.elapsed = t - state.startT;
    if (state.input === 'webcam') {
      state.lostT = present ? 0 : state.lostT + dt;
      if (state.lostT > 0.8) toast('Không thấy bạn trong camera!', 0.3);
    }
  }
  fly(dt, c, !playing || state.lostT > 0.8);
  if (playing) checkGates();
  updateCamera(dt);
  updateWind(dt);
  styleGates(gates, state.active, t);
  updateMarkers();
  updateHud(c);
  if (toastTimer > 0 && (toastTimer -= dt) <= 0) $('toast').classList.remove('show');
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
window.__sky = { state, camera, bird, gates };
resize();
requestAnimationFrame(frame);
