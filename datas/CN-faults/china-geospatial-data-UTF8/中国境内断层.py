import pygmt

# 创建一个新的 Figure 对象
fig = pygmt.Figure()

# 绘制海岸线
fig.coast(
    region="CN",  # 设置区域为中国,为ISO country code https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    projection="M15c",  # 设置投影为 Mercator 投影，宽度为 15 厘米
    frame="af",  # 设置框架和标注
    shorelines="0.5p,black",  # 绘制海岸线，线宽为 0.5 像素，颜色为黑色
    area_thresh=10000,  # 只绘制面积大于 10000 平方公里的陆地
)

# 绘制断层线
fig.plot(
    data="CN-border-L1.gmt",  # 断层线数据文件
    pen="1p,red"  # 线宽为 1 像素，颜色为红色
)
# fig.plot(
#     data="CN-block-L1.gmt",  # 断层线数据文件
#     pen="1p,red"  # 线宽为 1 像素，颜色为红色
# )
fig.plot(
    data="CN-block-L1.gmt",  # 断层线数据文件
    pen="1p,red" , # 线宽为 1 像素，颜色为红色
    fill = "blue@50"
)
# 显示图形
fig.show()
