from __future__ import annotations
import importlib.util as iu

import os
import platform

def get_rootPath(projectName: str = "geodesy"):
    status = iu.find_spec(projectName)
    if status is not None:
        return status.submodule_search_locations[0]
    else:
        try:
            os_name = platform.system()
            if os_name == "Windows":
                pathList = os.getcwd().split("\\")
                index = pathList.index(projectName)
                rootPath = ""
                for i in range(index + 1):
                    rootPath += pathList[i] + "\\"
            elif os_name == "Linux":
                pathList = os.getcwd().split("/")
                index = pathList.index(projectName)
                rootPath = ""
                for i in range(index + 1):
                    rootPath += pathList[i] + "/"
            else:
                rootPath = "."
                pass
            return rootPath
        except:
            raise FileNotFoundError("Please check your project name")

def chdir_rootPath():
    os.chdir(get_rootPath())

def get_datas_path(projectName:str = "geodesy"):
    rootPath = get_rootPath(projectName)
    tempResults = os.path.join(rootPath, "datas")
    return tempResults
