import subprocess

config_text = """
gmt begin CN-faults-labeling png
    # 设置中文字体配置文件 cidfmap 的目录，Windows 下无需此设置
    gmt set PS_CONVERT="C-I${HOME}/.gmt/"
    # GMT 处理中文存在一些已知 BUG
    # 需要设置 PS_CHAR_ENCODING 为 Standard 以绕过这一BUG
    gmt set PS_CHAR_ENCODING Standard
    gmt coast -JM10c -RTW -Baf -W0.5p,black
    # -aL="断层名称": set the "L" value (i.e., label) in segment headers using "断层名称"
    # :+Lh: take the label text from the "L" value in the segment header
    gmt convert CN-faults.gmt -aL="断层名称" | gmt plot -Sqn1:+Lh+f6p,39
gmt end
"""
subprocess.run(config_text, shell=True, check=True)
