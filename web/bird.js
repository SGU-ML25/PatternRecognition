// Con chim low-poly: thân hồng tím, cánh xanh lá có 2 khớp (trong/ngoài) để đập cánh.
import * as THREE from 'three';

function wingGeometry(len, rootChord, tipChord, sweep) {
  // hình thang trong mặt phẳng XZ, gốc tại x=0, kéo dài theo +X
  const s = new THREE.Shape();
  s.moveTo(0, -rootChord * 0.4);
  s.lineTo(len, -tipChord * 0.4 + sweep);
  s.lineTo(len * 0.92, tipChord * 0.6 + sweep);
  s.lineTo(0, rootChord * 0.6);
  s.closePath();
  const g = new THREE.ShapeGeometry(s);
  g.rotateX(Math.PI / 2);
  return g;
}

export class Bird {
  constructor() {
    this.root = new THREE.Group();          // vị trí + yaw
    this.body = new THREE.Group();          // pitch + roll
    this.root.add(this.body);
    const flat = (color, extra = {}) => new THREE.MeshLambertMaterial({ color, flatShading: true, ...extra });

    const torso = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), flat('#b8479a'));
    torso.scale.set(0.75, 0.7, 1.9);
    const belly = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), flat('#e48fc6'));
    belly.scale.set(0.6, 0.45, 1.5); belly.position.set(0, -0.3, 0.1);
    const head = new THREE.Mesh(new THREE.IcosahedronGeometry(0.55, 1), flat('#cf5aa8'));
    head.position.set(0, 0.35, -1.9);
    const beak = new THREE.Mesh(new THREE.ConeGeometry(0.16, 1.4, 5), flat('#f2c14e'));
    beak.rotation.x = -Math.PI / 2; beak.position.set(0, 0.25, -2.9);
    const eyeMat = new THREE.MeshBasicMaterial({ color: '#101010' });
    for (const sx of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.08, 6, 4), eyeMat);
      eye.position.set(sx * 0.35, 0.5, -2.2); this.body.add(eye);
    }
    const tailMat = flat('#57c27a', { side: THREE.DoubleSide });
    for (const a of [-0.35, 0, 0.35]) {
      const f = new THREE.Mesh(wingGeometry(1.8, 0.5, 0.7, 0), tailMat);
      f.rotation.y = Math.PI / 2 + a; f.position.set(0, 0.05, 1.6);
      this.body.add(f);
    }
    this.body.add(torso, belly, head, beak);

    // Cánh: pivot vai -> mảnh trong -> pivot khuỷu -> mảnh ngoài
    const innerMat = flat('#6bdc8c', { side: THREE.DoubleSide });
    const outerMat = flat('#3fae6a', { side: THREE.DoubleSide });
    this.wings = [];
    for (const side of [-1, 1]) {        // -1 = cánh trái (bên -X), 1 = cánh phải
      const shoulder = new THREE.Group();
      shoulder.position.set(side * 0.55, 0.25, -0.4);
      const inner = new THREE.Mesh(wingGeometry(2.4, 1.6, 1.4, 0.2), innerMat);
      const elbow = new THREE.Group();
      elbow.position.x = 2.3;
      const outer = new THREE.Mesh(wingGeometry(2.6, 1.4, 0.6, 0.7), outerMat);
      elbow.add(outer);
      const holder = new THREE.Group();     // lật cánh trái bằng scale.x = -1
      holder.scale.x = side;
      holder.add(inner, elbow);
      shoulder.add(holder);
      this.body.add(shoulder);
      this.wings.push({ side, shoulder, elbow, holder, angle: 0, fold: 0 });
    }
    this.phase = 0;
  }

  // up: góc nâng cánh (rad, >0 = giơ lên); fold: 0 = dang hết, 1 = khép
  _setWing(w, up, fold, dt) {
    const k = 1 - Math.exp(-14 * dt);
    w.angle += (up - w.angle) * k;
    w.fold += (fold - w.fold) * k;
    w.shoulder.rotation.z = w.side * w.angle;
    w.elbow.rotation.z = -w.angle * 0.35;      // phần ngoài trễ hơn -> trông mềm
    w.holder.rotation.y = -w.side * w.fold * 1.1; // gập cánh về phía sau khi bổ nhào
  }

  // mode 'mirror': cánh theo tay người chơi; 'auto': hoạt hoạ theo flap/dive
  animate(dt, { flap = 0, dive = 0, wingL = null, wingR = null }) {
    if (wingL !== null && wingR !== null) {
      const c = (a) => THREE.MathUtils.clamp(a, -1.1, 1.1);
      this._setWing(this.wings[0], c(wingL), dive, dt);
      this._setWing(this.wings[1], c(wingR), dive, dt);
      return;
    }
    this.phase += dt * (flap > 0.05 ? 14 : 2.2);
    const amp = flap > 0.05 ? 0.75 : 0.06;
    const up = 0.12 + Math.sin(this.phase) * amp;
    for (const w of this.wings) this._setWing(w, dive > 0.3 ? -0.2 : up, dive, dt);
  }
}
