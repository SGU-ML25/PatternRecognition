// Thế giới low-poly: địa hình, cây, bầu trời đêm, mây, cổng (gate).
import * as THREE from 'three';

// ---------- Noise có seed ----------
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function hash2(ix, iz) {
  let h = Math.imul(ix, 374761393) + Math.imul(iz, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function valueNoise(x, z) {
  const ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
  const u = fx * fx * (3 - 2 * fx), v = fz * fz * (3 - 2 * fz);
  const a = hash2(ix, iz), b = hash2(ix + 1, iz), c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
  return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v;
}
export function terrainHeight(x, z) {
  let h = 0, amp = 1, f = 1 / 420;
  for (let o = 0; o < 4; o++) { h += (valueNoise(x * f, z * f) - 0.5) * amp; amp *= 0.45; f *= 2.1; }
  return h * 110 + 8;
}

export const WORLD_SIZE = 5200;

export function buildWorld(scene) {
  const horizon = new THREE.Color('#33357a');
  scene.background = horizon;
  scene.fog = new THREE.Fog(horizon, 260, 1700);

  scene.add(new THREE.HemisphereLight('#8fa6ff', '#1e3b1a', 1.4));
  const moon = new THREE.DirectionalLight('#dfe6ff', 1.6);
  moon.position.set(-400, 600, -300);
  scene.add(moon);

  // Bầu trời gradient
  const sky = new THREE.Mesh(new THREE.SphereGeometry(3000, 32, 16), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color('#0d1238') }, bottom: { value: horizon } },
    vertexShader: 'varying vec3 p; void main(){ p = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 bottom; varying vec3 p; void main(){ gl_FragColor = vec4(mix(bottom, top, smoothstep(-0.05, 0.5, p.y)), 1.); }',
  }));
  sky.renderOrder = -1;
  scene.add(sky);

  // Sao + trăng (đi theo camera)
  const r = rng(7), starPos = [];
  for (let i = 0; i < 900; i++) {
    const th = r() * Math.PI * 2, ph = Math.acos(r() * 0.9 + 0.08);
    starPos.push(2600 * Math.sin(ph) * Math.cos(th), 2600 * Math.cos(ph), 2600 * Math.sin(ph) * Math.sin(th));
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: '#ffffff', size: 3, sizeAttenuation: false, fog: false }));
  const moonMesh = new THREE.Mesh(new THREE.SphereGeometry(70, 24, 12), new THREE.MeshBasicMaterial({ color: '#f3f0d8', fog: false }));
  moonMesh.position.set(-1300, 1100, -1900);
  const skyGroup = new THREE.Group();
  skyGroup.add(sky, stars, moonMesh);
  scene.add(skyGroup);

  // Địa hình low-poly
  const seg = 200;
  let geo = new THREE.PlaneGeometry(WORLD_SIZE, WORLD_SIZE, seg, seg);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, terrainHeight(pos.getX(i), pos.getZ(i)));
  geo = geo.toNonIndexed();
  const colors = [], cr = rng(3), c = new THREE.Color();
  const p2 = geo.attributes.position;
  for (let i = 0; i < p2.count; i += 3) {
    const y = (p2.getY(i) + p2.getY(i + 1) + p2.getY(i + 2)) / 3;
    const t = THREE.MathUtils.clamp((y + 40) / 120, 0, 1);
    c.setHSL(0.27 + cr() * 0.04, 0.55, 0.13 + t * 0.12 + cr() * 0.03);
    for (let k = 0; k < 3; k++) colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true })));

  // Cây: thân + tán (InstancedMesh)
  const N = 3200, tr = rng(11);
  const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.6, 0.9, 1, 5),
    new THREE.MeshLambertMaterial({ color: '#3a2a22', flatShading: true }), N);
  const crown = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0),
    new THREE.MeshLambertMaterial({ flatShading: true }), N);
  const palette = ['#2b56a8', '#3a74c9', '#24447f', '#1f5133', '#2f6b3a', '#6b3b33', '#2a3f73'].map((x) => new THREE.Color(x));
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
  for (let i = 0; i < N; i++) {
    const x = (tr() - 0.5) * WORLD_SIZE * 0.92, z = (tr() - 0.5) * WORLD_SIZE * 0.92;
    const y = terrainHeight(x, z), size = 4 + tr() * 6, hgt = 4 + tr() * 6;
    q.setFromEuler(new THREE.Euler(0, tr() * 6.28, 0));
    m.compose(p.set(x, y + hgt / 2, z), q, s.set(size / 6, hgt, size / 6)); trunk.setMatrixAt(i, m);
    m.compose(p.set(x, y + hgt + size * 0.6, z), q, s.set(size, size * (0.8 + tr() * 0.4), size)); crown.setMatrixAt(i, m);
    crown.setColorAt(i, palette[Math.floor(tr() * palette.length)]);
  }
  scene.add(trunk, crown);

  // Mây
  const cloudMat = new THREE.MeshLambertMaterial({ color: '#e8ecff', flatShading: true, emissive: '#3a3f70' });
  const cg = new THREE.IcosahedronGeometry(1, 0), clr = rng(21);
  for (let i = 0; i < 60; i++) {
    const g = new THREE.Group();
    for (let k = 0; k < 4 + Math.floor(clr() * 4); k++) {
      const b = new THREE.Mesh(cg, cloudMat);
      const sc = 10 + clr() * 16;
      b.scale.set(sc * 1.4, sc * 0.7, sc);
      b.position.set((clr() - 0.5) * 50, (clr() - 0.5) * 6, (clr() - 0.5) * 24);
      g.add(b);
    }
    g.position.set((clr() - 0.5) * WORLD_SIZE, 200 + clr() * 120, (clr() - 0.5) * WORLD_SIZE);
    scene.add(g);
  }
  return { skyGroup };
}

// ---------- Cổng ----------
export const GATE_RADIUS = 9;

export function buildCourse(scene, count = 16, seed = 42) {
  const r = rng(seed), gates = [];
  let yaw = 0, x = 0, z = -60;
  const ringGeo = new THREE.TorusGeometry(GATE_RADIUS, 1.0, 10, 40);
  const discGeo = new THREE.CircleGeometry(GATE_RADIUS * 0.28, 24);
  for (let i = 0; i < count; i++) {
    yaw += (r() - 0.5) * 1.3;
    // kéo đường bay về giữa bản đồ nếu đi quá xa
    const d = Math.hypot(x, z);
    if (d > 1300) {
      const toC = Math.atan2(x, z); // yaw hướng về tâm: forward = (-sin, -cos)
      yaw += Math.atan2(Math.sin(toC - yaw), Math.cos(toC - yaw)) * 0.6;
    }
    const dist = 170 + r() * 80;
    x += -Math.sin(yaw) * dist; z += -Math.cos(yaw) * dist;
    const y = Math.max(terrainHeight(x, z), terrainHeight(x + 15, z), terrainHeight(x - 15, z)) + 22 + r() * 26;

    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = yaw;
    const mat = new THREE.MeshStandardMaterial({ color: '#f6c33b', emissive: '#f6a21b', emissiveIntensity: 0.6, transparent: true });
    const ring = new THREE.Mesh(ringGeo, mat);
    const disc = new THREE.Mesh(discGeo, new THREE.MeshBasicMaterial({ color: '#fff3c4', transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
    const ground = terrainHeight(x, z);
    const poleH = y - GATE_RADIUS - ground;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, poleH, 6), mat);
    pole.position.y = -GATE_RADIUS - poleH / 2;
    g.add(ring, disc, pole);
    scene.add(g);
    gates.push({ group: g, mat, disc, pos: g.position, index: i, passed: false });
  }
  return gates;
}

export function styleGates(gates, active, t) {
  for (const g of gates) {
    if (g.passed) { g.group.visible = false; continue; }
    g.group.visible = true;
    const isActive = g.index === active;
    const pulse = isActive ? 1 + Math.sin(t * 5) * 0.06 : 1;
    g.group.scale.setScalar(pulse);
    g.mat.opacity = isActive ? 1 : 0.35;
    g.mat.emissiveIntensity = isActive ? 0.9 + Math.sin(t * 5) * 0.3 : 0.2;
    g.disc.visible = isActive;
  }
}
