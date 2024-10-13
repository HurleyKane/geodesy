# 该库对C++进行
# 安装 python setup.py bdist_wheel
# pip install dist/geodesy.whl
import setuptools

VERSION = '0.0.0'
package_name = f"geodesy"

setuptools.setup(
    name=package_name,
    version=VERSION,  # 两个地方都可以
    description="大地测量相关库",
    author="chenmingkai",
    author_email="<EMAIL>",
    url="https://github.com/hurleykane/geodesy",
    packages=setuptools.find_packages("."), # 自动找
    package_data={
        # 引入任何包下的pyd文件，加入字典则对应包下的文件
        "geodesy": ["*.pyd"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "xarray",
        "utm",
        "scipy",
        "pyproj",
        "windows-curses" ## 解决网址中的问题https://blog.csdn.net/S_o_l_o_n/article/details/106129004?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-106129004-blog-104267106.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-106129004-blog-104267106.235%5Ev43%5Epc_blog_bottom_relevance_base8&utm_relevant_index=6
    ],
    setup_requires=[
    ], # 用于指定在构建或安装项目之前所需要的依赖项。这些依赖通常是为了支持 setup.py 的运行，或者是构建包的工具依赖
    extras_require={
    },
)
