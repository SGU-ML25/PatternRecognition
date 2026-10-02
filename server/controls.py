"""Phương án 1: dùng hình học trên các khớp để suy ra lệnh điều khiển chim.

Toạ độ ảnh: x sang phải, y xuống dưới, ảnh KHÔNG lật gương (khớp "left_*" của người chơi
nằm bên phải ảnh). Mọi khoảng cách chuẩn hoá theo độ rộng vai S nên không phụ thuộc
người chơi đứng xa hay gần camera.
"""
import math
from collections import deque

import numpy as np

NOSE, LS, RS, LE, RE, LW, RW, LH, RH = 0, 5, 6, 7, 8, 9, 10, 11, 12


def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


class FlightControls:
    def __init__(self, conf_thr=0.3, bank_full_deg=40.0):
        self.conf_thr = conf_thr
        self.bank_full = math.radians(bank_full_deg)
        self.hist = deque(maxlen=8)  # (t, độ nâng tay trung bình) để tính vận tốc vỗ cánh
        self.flap = 0.0

    def _ok(self, kp, *ids):
        return all(kp[i, 2] > self.conf_thr for i in ids)

    def _arm_point(self, kp, wrist, elbow):
        """Đầu mút cánh: cổ tay nếu thấy, không thì khuỷu tay."""
        if self._ok(kp, wrist):
            return kp[wrist, :2]
        if self._ok(kp, elbow):
            return kp[elbow, :2]
        return None

    def __call__(self, kp, t):
        res = {"present": False, "bank": 0.0, "flap": 0.0, "dive": 0.0, "spread": 0.0,
               "wing_left": 0.0, "wing_right": 0.0, "arms_up": 0.0}
        if not self._ok(kp, LS, RS):
            self.hist.clear()
            self.flap *= 0.8
            return res
        ls, rs = kp[LS, :2], kp[RS, :2]
        S = float(np.linalg.norm(ls - rs)) + 1e-6
        mid = (ls + rs) / 2
        lt, rt = self._arm_point(kp, LW, LE), self._arm_point(kp, RW, RE)
        res["present"] = True
        if lt is None or rt is None:
            return res

        # 1) Rẽ: góc nghiêng của đường nối 2 đầu cánh (như máy bay nghiêng cánh).
        #    Tay trái của người chơi thấp hơn tay phải -> roll > 0 -> rẽ trái.
        roll = math.atan2(lt[1] - rt[1], lt[0] - rt[0])
        res["bank"] = _clamp(roll / self.bank_full)

        # 2) Góc từng cánh (để cánh chim trong game bắt chước tay người chơi), >0 = giơ lên
        res["wing_left"] = math.atan2(ls[1] - lt[1], abs(lt[0] - ls[0]) + 1e-6)
        res["wing_right"] = math.atan2(rs[1] - rt[1], abs(rt[0] - rs[0]) + 1e-6)

        # 3) Độ dang tay: khoảng cách ngang 2 đầu cánh / S (dang hết ≈ 4-5, khép ≈ 1)
        spread = abs(lt[0] - rt[0]) / S
        res["spread"] = spread

        # 4) Bổ nhào: tay khép sát người và thấp hơn vai
        below = ((lt[1] + rt[1]) / 2 - mid[1]) / S
        res["dive"] = _clamp((2.2 - spread) / 1.2, 0, 1) * _clamp(below / 0.5, 0, 1)

        # 5) Vỗ cánh: vận tốc đi xuống của tay (S/giây), chỉ tính nhịp đập xuống
        elev = float(((ls[1] - lt[1]) + (rs[1] - rt[1])) / 2 / S)
        res["arms_up"] = elev
        self.hist.append((t, elev))
        if len(self.hist) >= 3:
            t0, e0 = self.hist[0]
            v = -(elev - e0) / max(t - t0, 1e-3)  # > 0 khi tay đi xuống
            target = _clamp((v - 1.0) / 4.0, 0, 1) if spread > 1.8 else 0.0
        else:
            target = 0.0
        self.flap = max(target, self.flap * 0.85)  # giữ xung lực một lúc cho mượt
        res["flap"] = self.flap
        return {k: (v if isinstance(v, bool) else round(float(v), 4)) for k, v in res.items()}
