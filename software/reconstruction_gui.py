"""Local reconstruction, three-host fusion and interactive cloud/track replay."""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from qt_runtime import configure_qt
configure_qt()

from reconstruction import (Session, reconstruct_session, merge_reconstructions, read_rows)
from stereo_geometry import PassiveStereo


class CloudView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560,360)
        self.xyz=np.empty((0,3))
        self.colors=np.empty((0,3),np.uint8)
        self.markers=[]
        self.trails={}
        self.yaw=self.pitch=0.
        self.zoom=1.
        self.center=None
        self.last=None
        self.radius=200.
        self.show_ids=False

    def set_frame(self,xyz,colors,markers,trails):
        self.xyz,self.colors,self.markers,self.trails=xyz,colors,markers,trails
        focus=xyz if len(xyz) else np.array([m['xyz_mm'] for m in markers]).reshape(-1,3)
        if self.center is None and len(focus):
            self.center=np.median(focus,axis=0)
            self.radius=max(50.,float(np.percentile(np.linalg.norm(focus-self.center,axis=1),90)))
        self.update()

    def project(self,points):
        a,b=self.yaw,self.pitch
        R=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        S=np.array([[1,0,0],[0,math.cos(b),-math.sin(b)],[0,math.sin(b),math.cos(b)]])
        xyz=(np.asarray(points).reshape(-1,3)-(self.center if self.center is not None else 0))@(S@R).T
        scale=min(self.width(),self.height())*.42/self.radius*self.zoom
        return xyz[:,:2]*scale+[self.width()/2,self.height()/2],xyz[:,2]

    def paintEvent(self,event):
        p=QtGui.QPainter(self)
        p.fillRect(self.rect(),QtGui.QColor('#161a1d'))
        if len(self.xyz):
            step=max(1,len(self.xyz)//12000)
            xy,z=self.project(self.xyz[::step])
            colors=self.colors[::step]
            for i in np.argsort(-z):
                if 0<=xy[i,0]<self.width() and 0<=xy[i,1]<self.height():
                    p.setPen(QtGui.QPen(QtGui.QColor(*map(int,colors[i])),2))
                    p.drawPoint(QtCore.QPointF(*xy[i]))
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor('#efbb65'),1))
        for values in self.trails.values():
            if len(values)>1:
                xy,_=self.project(values)
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*q) for q in xy]))
        for mark in self.markers:
            xy,_=self.project([mark['xyz_mm']])
            pos=QtCore.QPointF(*xy[0])
            p.setBrush(QtGui.QColor('#f5d189'))
            p.drawEllipse(pos,3,3)
            if self.show_ids:
                p.drawText(pos+QtCore.QPointF(5,-4),mark['id'])
        p.setPen(QtGui.QColor('#dddddd'))
        p.drawText(12,22,'拖动旋转 · 滚轮缩放 · 双击重置')

    def mousePressEvent(self,event): self.last=event.pos()
    def mouseMoveEvent(self,event):
        if self.last is not None:
            delta=event.pos()-self.last
            self.yaw+=delta.x()*.008
            self.pitch+=delta.y()*.008
            self.last=event.pos()
            self.update()
    def mouseReleaseEvent(self,event): self.last=None
    def mouseDoubleClickEvent(self,event):
        self.yaw=self.pitch=0.
        self.zoom=1.
        self.center=None
        self.set_frame(self.xyz,self.colors,self.markers,self.trails)
    def wheelEvent(self,event):
        self.zoom=float(np.clip(self.zoom*math.exp(event.angleDelta().y()/1200),.1,20))
        self.update()


class Job(QtCore.QThread):
    message=QtCore.pyqtSignal(str)
    done=QtCore.pyqtSignal(str)
    failed=QtCore.pyqtSignal(str)
    def __init__(self,fn,args):
        super().__init__()
        self.fn,self.args=fn,args
    def run(self):
        try:
            result=self.fn(*self.args,progress=self.message.emit,cancel=self.isInterruptionRequested)
            self.done.emit(str(result))
        except Exception as exc:
            import logging
            logging.exception('reconstruction failed')
            self.failed.emit(str(exc))


class ReconstructionWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('D435i 本机重建 / 三机音视频合成 / 三维回放')
        self.resize(1220,880)
        self.job=None
        self.result=None
        self.frames=[]
        self.tabs=QtWidgets.QTabWidget()
        root=QtWidgets.QVBoxLayout()
        root.addWidget(self.tabs)
        local=QtWidgets.QWidget()
        layout=QtWidgets.QVBoxLayout(local)
        self.source=self.path_row(layout,'本机 session 目录')
        self.near=self.spin(100,2000,300)
        self.far=self.spin(200,5000,1800)
        self.threshold=self.spin(50,254,210)
        self.step=self.spin(1,60,1)
        line=QtWidgets.QHBoxLayout()
        for title,widget in [('最近 mm',self.near),('最远 mm',self.far),('反光阈值',self.threshold),('每 N 帧',self.step)]:
            line.addWidget(QtWidgets.QLabel(title)); line.addWidget(widget)
        layout.addLayout(line)
        self.roi=[self.spin(0,4096,0) for _ in range(4)]
        line=QtWidgets.QHBoxLayout()
        for title,widget in zip(('ROI x','y','宽(0=全图)','高(0=全图)'),self.roi):
            line.addWidget(QtWidgets.QLabel(title));line.addWidget(widget)
        preview=QtWidgets.QPushButton('显示校正后首帧')
        preview.clicked.connect(self.preview)
        line.addWidget(preview)
        layout.addLayout(line)
        self.preview_label=QtWidgets.QLabel('ROI 坐标按校正后的 IR1 设置；可框住人脸以排除背景亮点。')
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        layout.addWidget(self.preview_label)
        local_go=QtWidgets.QPushButton('选择新输出目录并重建本机三路画面')
        local_go.clicked.connect(self.local_run)
        layout.addWidget(local_go)
        self.tabs.addTab(local,'本机重建')

        merge=QtWidgets.QWidget()
        layout=QtWidgets.QVBoxLayout(merge)
        self.merge_paths=[];self.session_paths=[];self.anchors=[]
        for i in range(3):
            self.merge_paths.append(self.path_row(layout,f'机位 {i+1} 本机重建目录'))
            self.session_paths.append(self.path_row(layout,f'机位 {i+1} 原始 session（搬迁后填写）'))
            row=QtWidgets.QHBoxLayout()
            a,b=self.spin(0,10000000,0),self.spin(0,10000000,1)
            self.anchors.append((a,b))
            row.addWidget(QtWidgets.QLabel('共同事件 1 / 2：重建帧索引'))
            row.addWidget(a);row.addWidget(b);layout.addLayout(row)
        self.use_anchors=QtWidgets.QCheckBox('用两个共同视频事件校正视频时间（音轨仍自动匹配）')
        layout.addWidget(self.use_anchors)
        self.tolerance=self.spin(1,50,10)
        row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel('允许帧时间残差 ms'))
        row.addWidget(self.tolerance);layout.addLayout(row)
        note=QtWidgets.QLabel('按每帧保存的本机位姿戳完成三机结算，无需录制前生成或加载三机结算文件。\n'
            '合成包含九路原始视频、三条原始音轨、三条对齐音轨和三维点云轨迹。\n'
            '音频相关仅辅助时间估计；不能证明同步线连接或曝光同步。复制原始数据需要相应磁盘空间。')
        note.setWordWrap(True);layout.addWidget(note)
        merge_go=QtWidgets.QPushButton('按位姿戳结算、对齐音轨并合成三机结果')
        merge_go.clicked.connect(self.merge_run)
        layout.addWidget(merge_go)
        self.tabs.addTab(merge,'三机合成')

        replay=QtWidgets.QWidget();layout=QtWidgets.QVBoxLayout(replay)
        row=QtWidgets.QHBoxLayout()
        load=QtWidgets.QPushButton('打开三维结果')
        load.clicked.connect(self.pick_result)
        self.play=QtWidgets.QPushButton('播放 / 暂停')
        self.play.clicked.connect(self.toggle_play)
        self.smooth=QtWidgets.QCheckBox('平滑与短时补全');self.smooth.setChecked(True)
        self.smooth.toggled.connect(lambda _:self.show_frame(self.slider.value()))
        for w in (load,self.play,self.smooth):row.addWidget(w)
        layout.addLayout(row)
        self.cloud=CloudView();layout.addWidget(self.cloud,1)
        show_ids=QtWidgets.QCheckBox('显示点号')
        show_ids.toggled.connect(self.set_ids_visible)
        row.addWidget(show_ids)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame);layout.addWidget(self.slider)
        self.frame_label=QtWidgets.QLabel('未载入');layout.addWidget(self.frame_label)
        self.tabs.addTab(replay,'三维回放')
        self.timer=QtCore.QTimer(self);self.timer.setInterval(15);self.timer.timeout.connect(self.tick)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(95)
        root.addWidget(self.log)
        self.cancel=QtWidgets.QPushButton('取消处理');self.cancel.clicked.connect(self.cancel_job)
        root.addWidget(self.cancel)
        body=QtWidgets.QWidget();body.setLayout(root);self.setCentralWidget(body)
        self.processing_buttons=[local_go,merge_go]

    @staticmethod
    def spin(low,high,value):
        w=QtWidgets.QSpinBox();w.setRange(low,high);w.setValue(value);return w

    def set_ids_visible(self,enabled):
        self.cloud.show_ids=enabled
        self.cloud.update()

    def path_row(self,layout,label):
        row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel(label))
        edit=QtWidgets.QLineEdit();row.addWidget(edit,1)
        pick=QtWidgets.QPushButton('选择')
        def choose():
            path=QtWidgets.QFileDialog.getExistingDirectory(self,label)
            if path:edit.setText(path)
        pick.clicked.connect(choose);row.addWidget(pick);layout.addLayout(row)
        return edit

    def new_output(self):
        # A new folder is required so original or previous results cannot be overwritten.
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'选择输出所在的父目录')
        if not parent:return None
        name,ok=QtWidgets.QInputDialog.getText(self,'新结果目录','目录名',text='reconstruction_'+time.strftime('%Y%m%d_%H%M%S'))
        if not ok or not name:return None
        if Path(name).name!=name or name in ('.','..'):
            raise ValueError('请只输入新目录名')
        return str(Path(parent)/name)

    def preview(self):
        session=None
        try:
            session=Session(self.source.text())
            j=session.nearest('ir2',session.rows['ir1'][0],2.)
            if j is None:raise ValueError('首帧没有匹配的 IR2')
            stereo=PassiveStereo(session.meta)
            gray=stereo.rectify(session.image('ir1',0),session.image('ir2',j))[0]
            image=QtGui.QImage(gray.data,gray.shape[1],gray.shape[0],gray.strides[0],QtGui.QImage.Format_Grayscale8).copy()
            self.preview_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(900,300,QtCore.Qt.KeepAspectRatio))
            self.log.appendPlainText(f'校正帧尺寸 {gray.shape[1]} × {gray.shape[0]}；ROI 使用原分辨率像素坐标')
        except Exception as exc:self.error(str(exc))
        finally:
            if session:session.close()

    def local_run(self):
        try:
            if self.near.value()>=self.far.value():raise ValueError('最近距离必须小于最远距离')
            output=self.new_output()
            if not output:return
            roi=[w.value() for w in self.roi]
            opts=dict(near_mm=self.near.value(),far_mm=self.far.value(),threshold=self.threshold.value(),
                      frame_step=self.step.value(),roi=roi if roi[2]>0 and roi[3]>0 else None)
            self.start_job(reconstruct_session,(self.source.text(),output,opts))
        except Exception as exc:self.error(str(exc))

    def merge_run(self):
        try:
            if any(not p.text().strip() for p in self.merge_paths):raise ValueError('请选择三份本机重建目录')
            output=self.new_output()
            if not output:return
            opts=dict(max_time_error_ms=self.tolerance.value())
            supplied=[p.text().strip() for p in self.session_paths]
            if any(supplied):
                if not all(supplied):raise ValueError('搬迁后需要填写全部三份原始 session 目录')
                opts['sessions']=supplied
            if self.use_anchors.isChecked():opts['video_anchors']=[(a.value(),b.value()) for a,b in self.anchors]
            self.start_job(merge_reconstructions,([p.text() for p in self.merge_paths],output,opts))
        except Exception as exc:self.error(str(exc))

    def start_job(self,fn,args):
        if self.job and self.job.isRunning():raise ValueError('已有处理任务，请等待或取消')
        self.timer.stop()
        self.job=Job(fn,args)
        self.job.message.connect(self.log.appendPlainText)
        self.job.failed.connect(self.error)
        self.job.done.connect(self.load_result)
        self.job.finished.connect(lambda:[b.setEnabled(True) for b in self.processing_buttons])
        for b in self.processing_buttons:b.setEnabled(False)
        self.job.start()

    def cancel_job(self):
        if self.job:self.job.requestInterruption()

    def error(self,message):
        self.log.appendPlainText(message)
        QtWidgets.QMessageBox.warning(self,'处理信息',message)

    def pick_result(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'选择含 result.json 的三维结果目录')
        if folder:self.load_result(folder)

    def load_result(self,folder):
        try:
            self.timer.stop()
            meta=json.loads((Path(folder)/'result.json').read_text(encoding='utf8'))
            if not meta.get('complete'):raise ValueError('结果未完成，不能作为完整回放')
            frames=read_rows(Path(folder)/'frames_3d.jsonl')
            if not frames:raise ValueError('结果没有帧')
            self.result=Path(folder);self.frames=frames;self.cloud.center=None
            self.slider.setRange(0,len(self.frames)-1);self.slider.setValue(0)
            self.show_frame(0);self.tabs.setCurrentIndex(2)
            self.log.appendPlainText('已加载：'+folder)
        except Exception as exc:self.error(str(exc))

    def show_frame(self,index):
        if not self.frames:return
        try:
            frame=self.frames[index]
            key='display_markers' if self.smooth.isChecked() else 'markers'
            xyz,color=np.empty((0,3)),np.empty((0,3),np.uint8)
            if frame.get('cloud'):
                with np.load(self.result/frame['cloud']) as data:xyz,color=data['xyz_mm'],data['rgb']
            trails={}
            for previous in self.frames[max(0,index-30):index+1]:
                for m in previous.get(key,previous['markers']):trails.setdefault(m['id'],[]).append(m['xyz_mm'])
            self.cloud.set_frame(xyz,color,frame.get(key,frame['markers']),trails)
            self.frame_label.setText(f'帧 {index} / {len(self.frames)-1} | {frame["time_ms"]:.1f} ms | '
                                     f'{len(xyz)} 表面点 | {len(frame["markers"])} Mark')
        except Exception as exc:
            self.timer.stop();self.log.appendPlainText('回放失败：'+str(exc))

    def toggle_play(self):
        if self.timer.isActive():self.timer.stop();return
        if not self.frames:return
        self.play_wall=time.monotonic();self.play_time=self.frames[self.slider.value()]['time_ms']
        self.timer.start()

    def tick(self):
        target=self.play_time+(time.monotonic()-self.play_wall)*1000
        times=[f['time_ms'] for f in self.frames]
        index=min(len(times)-1,max(0,int(np.searchsorted(times,target,side='right'))-1))
        self.slider.setValue(index)
        if index==len(times)-1:self.timer.stop()

    def closeEvent(self,event):
        if self.job and self.job.isRunning():
            self.job.requestInterruption()
            self.log.appendPlainText('正在取消处理，请完成后关闭窗口')
            event.ignore()
        else:
            self.timer.stop();event.accept()
