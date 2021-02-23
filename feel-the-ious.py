from sys import dont_write_bytecode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


from matplotlib import rcParams
from numpy.lib.arraysetops import intersect1d
config = {
    "font.family":'Monospace',
    "font.size": 10
}
rcParams.update(config)


def xyc2xylt(xc, yc, w, h):
    return (xc-w/2, yc-h/2)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.set_aspect(aspect='equal')#, anchor='W')
plt.subplots_adjust(right=0.6)

plt.axis([0, 1, 0, 1])
ax.xaxis.set_ticks_position('top')

axcolor = 'lightgoldenrodyellow'
sld_left = 0.7
sld_top = 0.8
sld_deltav = 0.03
sld_w = 0.22
sld_h = 0.02

# xc1, yc1, w1, h1 = 0.4, 0.5, 0.5, 0.5
# xc2, yc2, w2, h2 = 0.65, 0.6, 0.4, 0.4
xc1, yc1, w1, h1 = 0.5, 0.5, 0.5, 0.5
xc2, yc2, w2, h2 = 0.5, 0.5, 0.5, 0.5

ax_xcgt_sld = plt.axes([sld_left, sld_top, sld_w, sld_h], facecolor=axcolor)
ax_ycgt_sld = plt.axes([sld_left, sld_top-sld_deltav, sld_w, sld_h], facecolor=axcolor)
ax_wgt_sld = plt.axes([sld_left, sld_top-sld_deltav*2, sld_w, sld_h], facecolor=axcolor)
ax_hgt_sld = plt.axes([sld_left, sld_top-sld_deltav*3, sld_w, sld_h], facecolor=axcolor)
xcgt_sld = Slider(ax_xcgt_sld, 'xc_gt', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=xc1)
ycgt_sld = Slider(ax_ycgt_sld, 'yc_gt', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=yc1)
wgt_sld = Slider(ax_wgt_sld, 'w_gt', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=w1)
hgt_sld = Slider(ax_hgt_sld, 'h_gt', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=h1)

ax_xc1_sld = plt.axes([sld_left, sld_top-sld_deltav*5, sld_w, sld_h], facecolor=axcolor)
ax_yc1_sld = plt.axes([sld_left, sld_top-sld_deltav*6, sld_w, sld_h], facecolor=axcolor)
ax_w1_sld = plt.axes([sld_left, sld_top-sld_deltav*7, sld_w, sld_h], facecolor=axcolor)
ax_h1_sld = plt.axes([sld_left, sld_top-sld_deltav*8, sld_w, sld_h], facecolor=axcolor)
xc1_sld = Slider(ax_xc1_sld, 'xc1', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=xc1)
yc1_sld = Slider(ax_yc1_sld, 'yc1', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=yc1)
w1_sld = Slider(ax_w1_sld, 'w1', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=w1)
h1_sld = Slider(ax_h1_sld, 'h1', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=h1)

ax_xc2_sld = plt.axes([sld_left, sld_top-sld_deltav*10, sld_w, sld_h], facecolor=axcolor)
ax_yc2_sld = plt.axes([sld_left, sld_top-sld_deltav*11, sld_w, sld_h], facecolor=axcolor)
ax_w2_sld = plt.axes([sld_left, sld_top-sld_deltav*12, sld_w, sld_h], facecolor=axcolor)
ax_h2_sld = plt.axes([sld_left, sld_top-sld_deltav*13, sld_w, sld_h], facecolor=axcolor)
xc2_sld = Slider(ax_xc2_sld, 'xc2', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=xc2)
yc2_sld = Slider(ax_yc2_sld, 'yc2', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=yc2)
w2_sld = Slider(ax_w2_sld, 'w2', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=w2)
h2_sld = Slider(ax_h2_sld, 'h2', 0.001, 1, valstep=0.01, valfmt='% .2f', valinit=h2)

ax_ious = plt.axes([sld_left-0.02, 0, sld_w, 0.3], )
plt.axis('off')

# IoU/GIoU/DIoU/CIoU计算，部分参考了[https://zhuanlan.zhihu.com/p/94799295]
def xywh2xyxy(box):
    x,y,w,h = box
    xlt = x - w/2
    ylt = y - h/2
    xrb = x + w/2
    yrb = y + h/2
    return xlt,ylt,xrb,yrb

def box_area(box):
    return box[2]*box[3]

def IoU(box1, box2):
    xlt1, ylt1, xrb1, yrb1 = xywh2xyxy(box1)
    xlt2, ylt2, xrb2, yrb2 = xywh2xyxy(box2)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xlt1, xlt2])
    yy1 = np.max([ylt1, ylt2])
    xx2 = np.min([xrb1, xrb2])
    yy2 = np.min([yrb1, yrb2])
    # 计算两个矩形框面积
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1])) #计算交集面积
    union_area = area1 + area2 - inter_area #计算并集面积
    iou = inter_area / union_area #计算交并比
    return iou, union_area, inter_area

def GIoU(box1, box2):
    #分别是第一个矩形左右上下的坐标
    xlt1, ylt1, xrb1, yrb1 = xywh2xyxy(box1)
    xlt2, ylt2, xrb2, yrb2 = xywh2xyxy(box2)
    iou, u, _ = IoU(box1, box2)
    xx1 = np.min([xlt1, xlt2])
    yy1 = np.min([ylt1, ylt2])
    xx2 = np.max([xrb1, xrb2])
    yy2 = np.max([yrb1, yrb2])
    ac = (yy2 - yy1) * (xx2 - xx1)
    giou = iou - np.abs(ac-u)/np.abs(ac)
    return giou

def DIoU(box1, box2):
    xlt1, ylt1, xrb1, yrb1 = xywh2xyxy(box1)
    xlt2, ylt2, xrb2, yrb2 = xywh2xyxy(box2)
    iou, _, _ = IoU(box1, box2)
    rou_2 = (box1[2]-box2[2])**2 + (box1[3]-box2[3])**2
    xx1 = np.min([xlt1, xlt2])
    yy1 = np.min([ylt1, ylt2])
    xx2 = np.max([xrb1, xrb2])
    yy2 = np.max([yrb1, yrb2])
    c_2 = (xx2-xx1)**2 + (yy2-yy1)**2
    diou = iou - rou_2/c_2
    return diou, iou

def CIoU(box1, box2):
    diou, iou = DIoU(box1, box2)
    v = (4/(np.pi**2)) * (np.arctan(box2[2]/box2[3]) - np.arctan(box1[2]/box1[3]))**2
    alpha = v / (1 - iou + v + 1e-6)
    ciou = diou - alpha*v
    return ciou, diou
# End


def calc_ious():
    boxgt = xcgt_sld.val, ycgt_sld.val, wgt_sld.val, hgt_sld.val
    box1 = xc1_sld.val, yc1_sld.val, w1_sld.val, h1_sld.val
    box2 = xc2_sld.val, yc2_sld.val, w2_sld.val, h2_sld.val
    iou1, union_area1, inter_area1 = IoU(box1, boxgt)
    giou1 = GIoU(box1, boxgt)
    ciou1, diou1 = CIoU(box1, boxgt)
    iou2, union_area2, inter_area2 = IoU(box2, boxgt)
    giou2 = GIoU(box2, boxgt)
    ciou2, diou2 = CIoU(box2, boxgt)
    return iou1, giou1, diou1, ciou1, iou2, giou2, diou2, ciou2


def show_ious():
    iou1, giou1, diou1, ciou1, iou2, giou2, diou2, ciou2 = calc_ious()
    plt.axis('off')
    # ax_ious.text(0, 1, 'area1 = % .4f'%area1)
    # ax_ious.text(0, 0.9, 'area2 = % .4f'%area2)
    # ax_ious.text(0, 0.8, 'inter = % .4f'%inter_area)
    # ax_ious.text(0, 0.7, 'union = % .4f'%union_area)
    ax_ious.text(0, 0.9, 'Loss')
    ax_ious.text(0.5, 0.9, ' BBox1', color='r')
    ax_ious.text(0.9, 0.9, ' BBox2', color='b')
    ax_ious.text(0, 0.8, '1-IoU')
    ax_ious.text(0.5, 0.8, '%.4f'%(1-iou1), color='r')
    ax_ious.text(0.9, 0.8, '%.4f'%(1-iou2), color='b')
    ax_ious.text(0, 0.7, '1-GIoU')
    ax_ious.text(0.5, 0.7, '%.4f'%(1-giou1), color='r')
    ax_ious.text(0.9, 0.7, '%.4f'%(1-giou2), color='b')
    ax_ious.text(0, 0.6, '1-DIoU')
    ax_ious.text(0.5, 0.6, '%.4f'%(1-diou1), color='r')
    ax_ious.text(0.9, 0.6, '%.4f'%(1-diou2), color='b')
    ax_ious.text(0, 0.5, '1-CIoU')
    ax_ious.text(0.5, 0.5, '%.4f'%(1-ciou1), color='r')
    ax_ious.text(0.9, 0.5, '%.4f'%(1-ciou2), color='b')


changed_slider = ''
def mark_changed_slider(slider):
    def func(val):
        global changed_slider
        changed_slider = slider
    return func

def set_val_when_lt0(xy, wh):
    if changed_slider.val == xy:
        changed_slider.set_val(wh/2)
    elif changed_slider.val == wh:
        changed_slider.set_val(xy*2)

def set_val_when_gt0(xy, wh):
    if changed_slider.val == xy:
        changed_slider.set_val(1-wh/2)
    elif changed_slider.val == wh:
        changed_slider.set_val((1-xy)*2)

def update(val):
    xcgt, ycgt, wgt, hgt = xcgt_sld.val, ycgt_sld.val, wgt_sld.val, hgt_sld.val
    xc1, yc1, w1, h1 = xc1_sld.val, yc1_sld.val, w1_sld.val, h1_sld.val
    xc2, yc2, w2, h2 = xc2_sld.val, yc2_sld.val, w2_sld.val, h2_sld.val
    
    # 令bbox不能超出边界
    if xcgt - wgt/2 < 0: set_val_when_lt0(xcgt, wgt)
    if xcgt + wgt/2 > 1: set_val_when_gt0(xcgt, wgt)
    if ycgt - hgt/2 < 0: set_val_when_lt0(ycgt, hgt)
    if ycgt + hgt/2 > 1: set_val_when_gt0(ycgt, hgt)
    if xc1 - w1/2 < 0: set_val_when_lt0(xc1, w1)
    if xc1 + w1/2 > 1: set_val_when_gt0(xc1, w1)
    if yc1 - h1/2 < 0: set_val_when_lt0(yc1, h1)
    if yc1 + h1/2 > 1: set_val_when_gt0(yc1, h1)
    if xc2 - w2/2 < 0: set_val_when_lt0(xc2, w2)
    if xc2 + w2/2 > 1: set_val_when_gt0(xc2, w2)
    if yc2 - h2/2 < 0: set_val_when_lt0(yc2, h2)
    if yc2 + h2/2 > 1: set_val_when_gt0(yc2, h2)
    
    ax.clear()
    show_boxes()
    ax_ious.clear()
    show_ious()
    fig.canvas.draw_idle()


slds = [xcgt_sld, ycgt_sld, wgt_sld, hgt_sld, xc1_sld, yc1_sld, w1_sld, h1_sld, xc2_sld, yc2_sld, w2_sld, h2_sld]
for sld in slds:
    sld.on_changed(mark_changed_slider(sld))
    sld.on_changed(update)

def show_boxes():
    xcgt, ycgt, wgt, hgt = xcgt_sld.val, ycgt_sld.val, wgt_sld.val, hgt_sld.val
    xc1, yc1, w1, h1 = xc1_sld.val, yc1_sld.val, w1_sld.val, h1_sld.val
    xc2, yc2, w2, h2 = xc2_sld.val, yc2_sld.val, w2_sld.val, h2_sld.val
    rectgt = plt.Rectangle(xyc2xylt(xcgt, ycgt, wgt, hgt), wgt, hgt, alpha=0.5, fill=True, color='g', linewidth=2)
    rect1 = plt.Rectangle(xyc2xylt(xc1, yc1, w1, h1), w1, h1, alpha=0.6, fill=True, color='r', linewidth=2)
    rect2 = plt.Rectangle(xyc2xylt(xc2, yc2, w2, h2), w2, h2, alpha=0.6, fill=True, color='b', linewidth=2)
    ax.add_patch(rectgt)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.invert_yaxis()

show_boxes()
show_ious()
plt.show()