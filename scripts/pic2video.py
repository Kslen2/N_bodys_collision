import os
import threading
import time
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

try:
	from PIL import Image, ImageTk
except Exception:
	Image = None
	ImageTk = None

try:
	import cv2
	import numpy as np
except Exception:
	cv2 = None
	np = None


class Pic2VideoApp:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("图片拼接视频工具")
		self.root.geometry("900x600")

		# 状态
		self.image_paths: List[str] = []
		# 预设默认路径：data 与 animation_saver
		try:
			base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		except Exception:
			base_dir = os.getcwd()
		self.default_data_dir = os.path.join(base_dir, "data")
		self.default_animation_dir = os.path.join(self.default_data_dir, "animation_saver")
		try:
			os.makedirs(self.default_animation_dir, exist_ok=True)
		except Exception:
			pass
		self.output_dir: Optional[str] = self.default_animation_dir
		self.output_filename: tk.StringVar = tk.StringVar(value="output.mp4")
		self.fps_var: tk.StringVar = tk.StringVar(value="12")
		self.keep_aspect_var: tk.BooleanVar = tk.BooleanVar(value=True)
		self.use_first_size_var: tk.BooleanVar = tk.BooleanVar(value=True)
		self.target_width_var: tk.StringVar = tk.StringVar(value="")
		self.target_height_var: tk.StringVar = tk.StringVar(value="")

		# 预览
		self.preview_running: bool = False
		self.preview_index: int = 0
		self.preview_target_size: Optional[Tuple[int, int]] = None
		self.preview_photo: Optional[ImageTk.PhotoImage] = None

		self._build_ui()

		# 初始化输出目录标签为默认 animation_saver
		try:
			if self.output_dir:
				self.output_dir_label.config(text=self.output_dir)
		except Exception:
			pass

	def _build_ui(self):
		# 左侧：图片列表与操作
		left_frame = tk.Frame(self.root)
		left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

		tk.Label(left_frame, text="已选择的图片(按顺序)").pack(anchor=tk.W)
		self.listbox = tk.Listbox(left_frame, height=20, selectmode=tk.EXTENDED)
		self.listbox.pack(fill=tk.BOTH, expand=True)

		# 图片计数标签与选择事件绑定
		self.count_label = tk.Label(left_frame, text="当前图片数：0（选中：0）", anchor=tk.W)
		self.count_label.pack(fill=tk.X, pady=2)
		self.listbox.bind("<<ListboxSelect>>", lambda e: self._update_image_count_label())

		btn_frame = tk.Frame(left_frame)
		btn_frame.pack(fill=tk.X, pady=5)
		tk.Button(btn_frame, text="添加图片", command=self.select_images).pack(side=tk.LEFT, padx=5)
		tk.Button(btn_frame, text="移除选中", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
		tk.Button(btn_frame, text="清空列表", command=self.clear_list).pack(side=tk.LEFT, padx=5)

		# 右侧：预览、设置、输出
		right_frame = tk.Frame(self.root)
		right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

		# 预览区域
		preview_group = tk.LabelFrame(right_frame, text="预览")
		preview_group.pack(fill=tk.BOTH, expand=True)

		self.preview_label = tk.Label(preview_group, bg="#222", width=60, height=20)
		self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

		preview_controls = tk.Frame(preview_group)
		preview_controls.pack(fill=tk.X)
		tk.Button(preview_controls, text="开始预览", command=self.start_preview).pack(side=tk.LEFT, padx=5)
		tk.Button(preview_controls, text="停止预览", command=self.stop_preview).pack(side=tk.LEFT, padx=5)

		# 设置区域
		settings_group = tk.LabelFrame(right_frame, text="设置")
		settings_group.pack(fill=tk.X, pady=10)

		tk.Label(settings_group, text="帧率(FPS)：").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
		tk.Entry(settings_group, textvariable=self.fps_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

		tk.Label(settings_group, text="输出文件名：").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
		tk.Entry(settings_group, textvariable=self.output_filename, width=20).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

		# 目标分辨率
		tk.Checkbutton(settings_group, text="使用首张图片尺寸", variable=self.use_first_size_var, command=self._toggle_target_size_inputs).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
		tk.Label(settings_group, text="目标分辨率(W×H)：").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
		self.width_entry = tk.Entry(settings_group, textvariable=self.target_width_var, width=10)
		self.height_entry = tk.Entry(settings_group, textvariable=self.target_height_var, width=10)
		self.width_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
		self.height_entry.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)

		tk.Checkbutton(settings_group, text="保持纵横比并居中填充", variable=self.keep_aspect_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

		tk.Button(settings_group, text="选择输出目录", command=self.select_output_dir).grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
		self.output_dir_label = tk.Label(settings_group, text="未选择")
		self.output_dir_label.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)

		# 初始化分辨率输入状态
		self._toggle_target_size_inputs()

		# 生成
		action_frame = tk.Frame(right_frame)
		action_frame.pack(fill=tk.X)
		tk.Button(action_frame, text="生成视频", command=self.generate_video).pack(side=tk.LEFT, padx=5, pady=5)

		self.status_label = tk.Label(right_frame, text="就绪", anchor=tk.W)
		self.status_label.pack(fill=tk.X)

		# 初始化计数显示
		self._update_image_count_label()

	# ---------- 列表操作 ----------
	def select_images(self):
		# 默认打开到 data 文件夹
		init_dir = self.default_data_dir if os.path.isdir(getattr(self, "default_data_dir", "")) else os.getcwd()
		paths = filedialog.askopenfilenames(title="选择图片", initialdir=init_dir, filetypes=[
			("图片", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
			("所有文件", "*.*"),
		])
		if not paths:
			return

		# 保留用户选择顺序
		for p in paths:
			if p not in self.image_paths:
				self.image_paths.append(p)
				self.listbox.insert(tk.END, p)

		self.status_label.config(text=f"已添加 {len(paths)} 张图片，总计 {len(self.image_paths)}")
		self._update_image_count_label()

	def remove_selected(self):
		selection = list(self.listbox.curselection())
		if not selection:
			return
		selection.reverse()
		for idx in selection:
			path = self.listbox.get(idx)
			self.listbox.delete(idx)
			if path in self.image_paths:
				self.image_paths.remove(path)
		self.status_label.config(text=f"移除完成，总计 {len(self.image_paths)}")
		self._update_image_count_label()

	def clear_list(self):
		self.listbox.delete(0, tk.END)
		self.image_paths.clear()
		self.status_label.config(text="列表已清空")
		self._update_image_count_label()

	# ---------- 预览 ----------
	def start_preview(self):
		if not self._check_pillow_available():
			return
		if not self.image_paths:
			messagebox.showwarning("提示", "请先选择图片")
			return
		fps = self._get_fps()
		if fps is None:
			return

		# 目标尺寸：首图或自定义
		self.preview_target_size = self._get_target_size()
		if self.preview_target_size is None:
			messagebox.showerror("错误", "无法读取第一张图片尺寸")
			return

		self.preview_running = True
		self.preview_index = 0
		self.status_label.config(text="预览中…")
		self._preview_loop(int(1000 / max(fps, 1)))

	def stop_preview(self):
		self.preview_running = False
		self.status_label.config(text="预览已停止")

	def _preview_loop(self, delay_ms: int):
		if not self.preview_running:
			return
		if not self.image_paths:
			self.preview_running = False
			return
		img_path = self.image_paths[self.preview_index % len(self.image_paths)]
		target_size = self.preview_target_size
		img = self._load_and_fit_image(img_path, target_size, self.keep_aspect_var.get())
		if img is not None:
			display_size = self._get_preview_display_size(target_size)
			disp_img = img.resize(display_size, Image.LANCZOS)
			self.preview_photo = ImageTk.PhotoImage(disp_img)
			self.preview_label.config(image=self.preview_photo)

		self.preview_index = (self.preview_index + 1) % len(self.image_paths)
		self.root.after(delay_ms, lambda: self._preview_loop(delay_ms))

	# ---------- 输出目录 ----------
	def select_output_dir(self):
		d = filedialog.askdirectory(title="选择输出目录")
		if d:
			self.output_dir = d
			self.output_dir_label.config(text=d)
			self.status_label.config(text="输出目录已设置")

	# ---------- 生成视频 ----------
	def generate_video(self):
		if not self._check_cv2_available() or not self._check_pillow_available():
			return
		if not self.image_paths:
			messagebox.showwarning("提示", "请先选择图片")
			return
		fps = self._get_fps()
		if fps is None:
			return
		if not self.output_dir:
			messagebox.showwarning("提示", "请先选择输出目录")
			return
		filename = self.output_filename.get().strip()
		if not filename:
			messagebox.showwarning("提示", "请输入输出文件名")
			return
		if not filename.lower().endswith(".mp4"):
			filename += ".mp4"

		target_size = self._get_target_size()
		if target_size is None:
			messagebox.showerror("错误", "无法读取第一张图片尺寸")
			return

		out_path = os.path.join(self.output_dir, filename)

		# 后台线程写视频，避免UI卡顿
		def worker():
			try:
				self._write_video(out_path, target_size, fps, self.keep_aspect_var.get())
				self.status_label.config(text=f"视频生成完成：{out_path}")
				messagebox.showinfo("完成", f"视频已生成：\n{out_path}")
			except Exception as e:
				self.status_label.config(text="生成失败")
				messagebox.showerror("错误", f"生成失败：{e}")

		self.status_label.config(text="正在生成视频…")
		threading.Thread(target=worker, daemon=True).start()

	# ---------- 工具方法 ----------
	def _get_fps(self) -> Optional[float]:
		try:
			fps = float(self.fps_var.get())
			if fps <= 0:
				raise ValueError("fps must be > 0")
			return fps
		except Exception:
			messagebox.showwarning("提示", "帧率输入无效，请输入大于0的数字")
			return None

	def _get_target_size(self) -> Optional[Tuple[int, int]]:
		if not self.image_paths:
			return None
		if self.use_first_size_var.get():
			p = self.image_paths[0]
			try:
				img = Image.open(p)
				return img.size  # (width, height)
			except Exception:
				return None
		# 使用自定义分辨率
		try:
			w = int(self.target_width_var.get())
			h = int(self.target_height_var.get())
			if w <= 0 or h <= 0:
				raise ValueError
			return (w, h)
		except Exception:
			messagebox.showwarning("提示", "请输入有效的目标分辨率(正整数)")
			return None

	def _load_and_fit_image(self, path: str, target_size: Tuple[int, int], keep_aspect: bool) -> Optional[Image.Image]:
		try:
			img = Image.open(path).convert("RGB")
			if keep_aspect:
				img = self._fit_keep_aspect(img, target_size)
			else:
				img = img.resize(target_size, Image.LANCZOS)
			return img
		except Exception:
			return None

	def _fit_keep_aspect(self, img: Image.Image, target: Tuple[int, int]) -> Image.Image:
		tw, th = target
		iw, ih = img.size
		scale = min(tw / iw, th / ih)
		nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
		resized = img.resize((nw, nh), Image.LANCZOS)
		canvas = Image.new("RGB", (tw, th), (0, 0, 0))
		x = (tw - nw) // 2
		y = (th - nh) // 2
		canvas.paste(resized, (x, y))
		return canvas

	def _get_preview_display_size(self, target: Tuple[int, int]) -> Tuple[int, int]:
		# 根据预览容器大小缩放显示，但保持与目标分辨率相同的画面比例
		try:
			cont_w = self.preview_label.winfo_width()
			cont_h = self.preview_label.winfo_height()
		except Exception:
			cont_w, cont_h = 0, 0

		if cont_w < 50 or cont_h < 50:
			# 尚未布局完成时的兜底尺寸
			cont_w, cont_h = 640, 360

		tw, th = target
		scale = min(cont_w / tw, cont_h / th)
		dw, dh = max(1, int(tw * scale)), max(1, int(th * scale))
		return (dw, dh)

	def _write_video(self, out_path: str, target_size: Tuple[int, int], fps: float, keep_aspect: bool):
		if cv2 is None:
			raise RuntimeError("OpenCV未安装")
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		w, h = target_size
		writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
		if not writer.isOpened():
			raise RuntimeError("无法打开视频写入器，请检查输出路径或编码器")

		for i, p in enumerate(self.image_paths):
			img = self._load_and_fit_image(p, target_size, keep_aspect)
			if img is None:
				continue
			frame = np.array(img)[:, :, ::-1]  # RGB->BGR
			writer.write(frame)
			self.status_label.config(text=f"写入第 {i+1}/{len(self.image_paths)} 帧…")

		writer.release()

	def _update_image_count_label(self):
		try:
			total = len(self.image_paths)
			selected = len(self.listbox.curselection())
			self.count_label.config(text=f"当前图片数：{total}（选中：{selected}）")
		except Exception:
			pass

	def _check_pillow_available(self) -> bool:
		if Image is None or ImageTk is None:
			messagebox.showerror("缺少依赖", "缺少Pillow库，请先安装：pip install pillow")
			return False
		return True

	def _check_cv2_available(self) -> bool:
		if cv2 is None or np is None:
			messagebox.showerror("缺少依赖", "缺少OpenCV库，请先安装：pip install opencv-python")
			return False
		return True

	def _toggle_target_size_inputs(self):
		use_first = self.use_first_size_var.get()
		state = tk.DISABLED if use_first else tk.NORMAL
		try:
			self.width_entry.config(state=state)
			self.height_entry.config(state=state)
		except Exception:
			pass


def main():
	root = tk.Tk()
	app = Pic2VideoApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()

