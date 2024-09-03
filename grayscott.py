import os
from pathlib import Path
from collections import OrderedDict

import moderngl
from pyrr import Matrix44, Vector4
from moderngl_window import geometry
import moderngl_window as mglw
import imgui
from moderngl_window.integrations.imgui import ModernglWindowRenderer


import cv2
import numpy as np


class GrayScottSolver:
    def __init__(self, wnd: mglw.WindowConfig):
        self.wnd = wnd
        self.ctx = wnd.ctx

        self.xgrid = 256
        self.ygrid = 256
        self.xrange = 2.5
        self.yrange = 2.5
        self.ic_range = 2.5 / 4
        self.seed = 0

        self.quad_fs = geometry.quad_fs()
        self.program = wnd.load_program(
            vertex_shader="quad.vert", fragment_shader="quad_grayscott.frag"
        )

        self.reset()

        self.c_du = 2e-5
        self.c_dv = 1e-5
        self.c_f = 0.04
        self.c_k = 0.06

        self.presets = OrderedDict(
            [
                # name, (du, dv, f, k)
                ("Default", (2e-5, 1e-5, 0.04, 0.06)),
                ("Bacteria", (2e-5, 1e-5, 0.035, 0.065)),
                ("Coral", (2e-5, 1e-5, 0.060, 0.062)),
                ("Worms", (2e-5, 1e-5, 0.050, 0.065)),
                ("Zebrafish", (2e-5, 1e-5, 0.035, 0.060)),
                ("Sample 1", (2e-5, 1e-5, 0.024, 0.056)),
                ("Sample 2", (2e-5, 1e-5, 0.040, 0.059)),
                ("Sample 3", (2e-5, 1e-5, 0.020, 0.056)),
                ("Sample 4", (2e-5, 1e-5, 0.016, 0.050)),
                ("Alpha", (2e-5, 1e-5, 0.019, 0.050)),
                ("Beta", (2e-5, 1e-5, 0.019, 0.045)),
                ("Gamma", (2e-5, 1e-5, 0.024, 0.055)),
                ("Delta", (2e-5, 1e-5, 0.029, 0.055)),
                ("Epsilon", (2e-5, 1e-5, 0.024, 0.063)),
                ("Zeta", (2e-5, 1e-5, 0.024, 0.060)),
                ("Eta", (2e-5, 1e-5, 0.034, 0.063)),
                ("Theta", (2e-5, 1e-5, 0.04, 0.060)),
                ("Iota", (2e-5, 1e-5, 0.049, 0.060)),
                ("Kappa", (2e-5, 1e-5, 0.04, 0.063)),
                ("Lambda", (2e-5, 1e-5, 0.035, 0.066)),
                ("Mu", (2e-5, 1e-5, 0.05, 0.066)),
                ("R", (2e-5, 1e-5, 0.010, 0.040)),
                ("B", (2e-5, 1e-5, 0.03, 0.050)),
            ]
        )
        self.selected_preset = 0

        self.phase_texture = self.wnd.load_texture_2d("phase.png", flip_y=False)
        self.wnd.imgui.register_texture(self.phase_texture)

    def reset(self):
        u = np.ones((self.xgrid, self.ygrid), dtype=np.float32)
        v = np.zeros((self.xgrid, self.ygrid), dtype=np.float32)

        x_ic_l = int(self.xgrid / 2 - self.ic_range * self.xgrid / self.xrange / 2)
        x_ic_r = int(self.xgrid / 2 + self.ic_range * self.ygrid / self.yrange / 2)
        y_ic_l = int(self.ygrid / 2 - self.ic_range * self.xgrid / self.xrange / 2)
        y_ic_r = int(self.ygrid / 2 + self.ic_range * self.ygrid / self.yrange / 2)

        u[x_ic_l:x_ic_r, y_ic_l:y_ic_r] = 0.5
        v[x_ic_l:x_ic_r, y_ic_l:y_ic_r] = 0.25

        # add some noise
        np.random.seed(self.seed)
        u += np.random.rand(self.xgrid, self.ygrid) * 0.01
        v += np.random.rand(self.xgrid, self.ygrid) * 0.01

        data = np.zeros((self.xgrid, self.ygrid, 2), dtype=np.float32)
        data[:, :, 0] = u
        data[:, :, 1] = v

        self.active_slot = 0

        self.texture0 = self.ctx.texture((self.ygrid, self.xgrid), 2, data.tobytes(), dtype="f4")
        # self.texture0.filter = moderngl.NEAREST, moderngl.NEAREST
        self.texture0.repeat_x = True
        self.texture0.repeat_y = True

        self.texture1 = self.ctx.texture((self.ygrid, self.xgrid), 2, data.tobytes(), dtype="f4")
        # self.texture1.filter = moderngl.NEAREST, moderngl.NEAREST
        self.texture1.repeat_x = True
        self.texture1.repeat_y = True

        self.fbo0 = self.ctx.framebuffer(color_attachments=[self.texture0])
        self.fbo1 = self.ctx.framebuffer(color_attachments=[self.texture1])

    def step(self, dt):
        self.texture0.use(0)
        self.texture1.use(1)
        fbo = self.fbo1 if self.active_slot == 0 else self.fbo0
        fbo.use()

        self.program["texture0"].value = 0 if self.active_slot == 0 else 1

        self.program["c_du"].value = self.c_du
        self.program["c_dv"].value = self.c_dv
        self.program["c_f"].value = self.c_f
        self.program["c_k"].value = self.c_k

        self.program["dx"] = 1 / self.xgrid
        self.program["dy"] = 1 / self.ygrid
        self.program["hx"] = self.xrange / self.xgrid
        self.program["hy"] = self.yrange / self.ygrid
        self.program["dt"] = dt

        self.quad_fs.render(self.program)
        self.active_slot = 1 if self.active_slot == 0 else 0

        # debug: read the texture back to CPU
        # data = np.frombuffer(self.get_texture().read(), dtype=np.float32).reshape((self.xgrid, self.ygrid, 2))
        # print(data[:, :, 0].min(), data[:, :, 0].max(), data[:, :, 1].min(), data[:, :, 1].max())

    def get_texture(self):
        return self.texture0 if self.active_slot == 0 else self.texture1

    def render_ui(self):
        imgui.begin("Equation Parameters")

        imgui.image(self.phase_texture.glo, 200, 200)

        changed, self.selected_preset = imgui.combo("Preset", self.selected_preset, list(self.presets.keys()))
        if changed:
            self.c_du, self.c_dv, self.c_f, self.c_k = self.presets[list(self.presets.keys())[self.selected_preset]]
        _, self.c_du = imgui.input_float("Du", self.c_du, format="%.8f")
        if self.c_du < 0:
            self.c_du = 0
        _, self.c_dv = imgui.input_float("Dv", self.c_dv, format="%.8f")
        if self.c_dv < 0:
            self.c_dv = 0
        _, self.c_f = imgui.drag_float("F", self.c_f, 0.001, 0.0, 1.0)
        _, self.c_k = imgui.drag_float("K", self.c_k, 0.001, 0.0, 1.0)

        _, self.seed = imgui.input_int("Seed", self.seed)

        wnd_x, wnd_y = imgui.get_window_position()
        wnd_x += 8
        wnd_y += 27
        draw_list = imgui.get_window_draw_list()
        # X axis -> k 0.03 - 0.07
        # Y axis -> f 0.00 - 0.08
        xline = 0.03 + (self.c_k - 0.03) / 0.04 * 200
        yline = 200 - (self.c_f / 0.08 * 200)
        draw_list.add_line(wnd_x + xline, wnd_y, wnd_x + xline, wnd_y + 200, imgui.get_color_u32_rgba(1, 0, 0, 1))
        draw_list.add_line(wnd_x, wnd_y + yline, wnd_x + 200, wnd_y + yline, imgui.get_color_u32_rgba(1, 0, 0, 1))

        imgui.end()

    def dump_data(self):
        data = np.frombuffer(self.get_texture().read(), dtype=np.float32).reshape((self.xgrid, self.ygrid, 2))
        return data.copy()

class GrayScottRenderer:
    def __init__(self, wnd: mglw.WindowConfig):
        self.wnd = wnd
        self.ctx = wnd.ctx
        self.program = wnd.load_program(
            vertex_shader="quad.vert", fragment_shader="quad_gradient.frag"
        )
        self.legend_texture = self.ctx.texture(
            (256, 1), 3, np.zeros((256, 1, 3), dtype=np.uint8).tobytes()
        )
        self.legend_fbo = self.ctx.framebuffer(color_attachments=[self.legend_texture])
        wnd.imgui.register_texture(self.legend_texture)
        self.quad_fs = geometry.quad_fs()

        self.reset()

    def reset(self):
        self.color1 = Vector4([0.0, 0.0, 0.0, 0.0])
        self.color2 = Vector4([0.0, 1.0, 0.0, 0.3])
        self.color3 = Vector4([1.0, 1.0, 0.0, 0.35])
        self.color4 = Vector4([1.0, 0.0, 0.0, 0.4])
        self.color5 = Vector4([1.0, 1.0, 1.0, 0.6])

    def render(self, texture, fbo):
        texture.use(0)
        self.legend_fbo.use()
        self.program["texture0"].value = 0
        self.program["is_legend"].value = 1
        self.program["color1"].write(self.color1.astype("f4"))
        self.program["color2"].write(self.color2.astype("f4"))
        self.program["color3"].write(self.color3.astype("f4"))
        self.program["color4"].write(self.color4.astype("f4"))
        self.program["color5"].write(self.color5.astype("f4"))
        self.quad_fs.render(self.program)

        fbo.use()
        self.program["is_legend"].value = 0
        self.quad.render(self.program)

    def save(self, texture, filename):
        texture_out = self.ctx.texture(texture.size, 3)
        fbo = self.ctx.framebuffer(color_attachments=[texture_out])
        fbo.use()
        texture.use(0)
        self.program["texture0"].value = 0
        self.program["is_legend"].value = 0
        self.quad_fs.render(self.program)
        data = texture_out.read()
        data = np.frombuffer(data, dtype=np.uint8).reshape((texture.width, texture.height, 3))
        cv2.imwrite(filename, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

    def save_legend(self, filename):
        data = self.legend_texture.read()
        data = np.frombuffer(data, dtype=np.uint8).reshape((*self.legend_texture.size, 3))
        cv2.imwrite(filename, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

    def resize_quad(self, window_aspect, image_aspect):
        if image_aspect > window_aspect:
            quad_size = (2.0, 2.0 * window_aspect / image_aspect)
        else:
            quad_size = (2.0 * image_aspect / window_aspect, 2.0)
        self.quad = geometry.quad_2d(size=quad_size)

    def render_ui(self):
        imgui.begin("Color Map")

        imgui.image(self.legend_texture.glo, 256, 32)
        
        def color_edit(color, label):
            _, col = imgui.color_edit3(f"Color {label}", *color[:3])
            _, loc = imgui.slider_float(f"Location {label}", color[3], 0.0, 1.0)
            return Vector4([col[0], col[1], col[2], loc])

        if imgui.tree_node("Color Editor"):
            if imgui.button("Reset"):
                self.reset()
            self.color1 = color_edit(self.color1, "1")
            self.color2 = color_edit(self.color2, "2")
            self.color3 = color_edit(self.color3, "3")
            self.color4 = color_edit(self.color4, "4")
            self.color5 = color_edit(self.color5, "5")
            imgui.tree_pop()

        if imgui.button("Save Legend"):
            self.save_legend("legend.png")
        
        imgui.end()


class GrayScottScheduler:
    def __init__(self, wnd: mglw.WindowConfig):
        self.wnd = wnd
        self.solver = GrayScottSolver(wnd)
        self.renderer = GrayScottRenderer(wnd)

        self.paused = False
        self.tick_per_frame = 10
        self.ticks = 0
        self.t = 0.0
        self.use_target = True
        self.target_t = 20000.0
        self.dt = 1

        self.share_xy = True
        self.xgrid = 256
        self.ygrid = 256
        self.xrange = 2.5
        self.yrange = 2.5
        self.ic_range = 2.5 / 4

    def render(self, fbo):
        if not self.paused:
            for _ in range(self.tick_per_frame):
                self.solver.step(self.dt)
                self.t += self.dt
                self.ticks += 1

                if self.use_target and self.t >= self.target_t:
                    self.paused = True
                    break

        self.renderer.render(self.solver.get_texture(), fbo)


    def reset(self):
        self.solver.xgrid = self.xgrid
        self.solver.ygrid = self.ygrid
        self.solver.xrange = self.xrange
        self.solver.yrange = self.yrange

        self.solver.reset()
        self.renderer.resize_quad(self.wnd.wnd.aspect_ratio, self.yrange / self.xrange)
        self.t = 0
        self.ticks = 0
        self.dt_max = min((self.xrange / self.xgrid) ** 2 / (4 * self.solver.c_du), 1.0)
        self.dt = self.dt_max * 0.9

    def save(self, filename=None):
        if filename is None:
            preset_name = list(self.solver.presets.keys())[self.solver.selected_preset]
            filename = f"output/{preset_name}_t{int(self.t)}_s{self.xgrid}_dt{self.dt:.3f}"
        data = self.solver.dump_data()
        os.makedirs("output", exist_ok=True)
        np.save(f"{filename}.npy", data)
        self.renderer.save(self.solver.get_texture(), f"{filename}.png")

    def render_ui(self):
        imgui.begin("Scheduler")
        if imgui.button("Reset Simulation"):
            self.reset()
        _, self.share_xy = imgui.checkbox("Share X/Y", self.share_xy)
        if self.share_xy:
            _, self.xgrid = imgui.drag_int("Grid", self.xgrid, 1, 32, 1024)
            self.ygrid = self.xgrid
            _, self.xrange = imgui.drag_float("Range", self.xrange, 1, 0.5, 5.0)
            self.yrange = self.xrange
        else:
            _, (self.xgrid, self.ygrid) = imgui.drag_int2("Grid", self.xgrid, self.ygrid, 32, 1024)
            _, (self.xrange, self.yrange) = imgui.drag_float2("Range", self.xrange, self.yrange, 0.5, 5.0)
        _, self.ic_range = imgui.drag_float("IC Range", self.ic_range, 0.01, 0.0, min(self.xrange, self.yrange))

        imgui.separator()

        if self.paused:
            if imgui.button("Resume"):
                self.paused = False
        else:
            if imgui.button("Pause"):
                self.paused = True
        _, self.tick_per_frame = imgui.slider_int("Ticks per frame", self.tick_per_frame, 1, 100)
        _, self.use_target = imgui.checkbox("Pause at Time", self.use_target)
        if self.use_target:
            _, self.target_t = imgui.drag_float("Target Time", self.target_t, 1000, 0)
        _, self.dt = imgui.drag_float("Dt", self.dt, 0.01, 0.0001, self.dt_max, format="%.4f")

        imgui.text(f"Dt: {self.dt:.4f}")
        imgui.text(f"Time: {self.t:.2f}")
        imgui.text(f"Ticks: {self.ticks}")

        imgui.separator()

        if imgui.button("Save"):
            self.save()

        imgui.end()

        self.solver.render_ui()
        self.renderer.render_ui()


class GrayScottWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Gray Scott"
    resource_dir = (Path(__file__).parent / "assets").resolve()
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)

        self.scheduler = GrayScottScheduler(self)
        self.scheduler.reset()

    def render(self, time: float, frametime: float):
        self.ctx.clear()

        # First pass: Render morphed image to FBO
        self.scheduler.render(self.ctx.screen)

        self.render_ui()

    def render_ui(self):
        imgui.new_frame()
        self.scheduler.render_ui()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)
        self.scheduler.renderer.resize_quad(
            self.wnd.aspect_ratio, self.scheduler.yrange / self.scheduler.xrange
        )

    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)


if __name__ == "__main__":
    mglw.run_window_config(GrayScottWindow)
