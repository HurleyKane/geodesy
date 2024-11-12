import pygmt
import os

# 设置中文字体配置文件目录，Windows 下无需此设置
os.environ["PS_CONVERT"] = f"C-I{os.path.expanduser('~')}/.gmt"

# 设置字符编码以绕过中文处理的已知 BUG
os.environ["PS_CHAR_ENCODING"] = "Standard"

# 创建一个新的 Figure 对象
fig = pygmt.Figure()

# 绘制海岸线
fig.coast(
    region="TW",  # 设置区域为中国（TW 表示中国）
    projection="M10c",  # 设置投影为 Mercator 投影，宽度为 10 厘米
    frame="af",  # 设置框架和标注
    shorelines="0.5p,black"  # 绘制海岸线，线宽为 0.5 像素，颜色为黑色
)

# 处理并绘制断层线
# -aL="断层名称": 设置段头中的 "L" 值为 "断层名称"
# -Sqn1:+Lh+f6p,39: 绘制带标签的断层线，标签字体大小为 6 像素，颜色为 39
with open("CN-faults.gmt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith(">"):
            label = line.split("=")[-1].strip()
            fig.text(
                x=[float(x) for x in line.split()[1:3]],
                y=[float(y) for y in line.split()[1:3]],
                text=label,
                font="6p,39",
                justify="CT"
            )
        else:
            coords = [list(map(float, line.split()))]
            fig.plot(
                data=coords,
                pen="1p,red",
                label=f"+L{label}+f6p,39"
            )

# 显示图形
fig.show()
