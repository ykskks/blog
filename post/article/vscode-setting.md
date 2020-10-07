---
title: "VSCodeの設定メモ"
date: 2020-10-07
tags: ["環境構築"]

---

VSCodeの`settings.json`のメモを置いておく。

随時更新していこうと思う。

全般的な設定とPython, markdown, htmlの設定がそれぞれ少しずつ。

```
{
    // general
    "editor.cursorStyle": "line",
    "editor.minimap.enabled": false,
    "editor.formatOnSave": true,
    // "editor.codeActionsOnSave": {
    //     "source.organizeImports": true
    // },
    "editor.largeFileOptimizations": false,
    "workbench.colorTheme": "Material Theme Palenight High Contrast",
    "workbench.iconTheme": "material-icon-theme",
    "bracket-pair-colorizer-2.colors": [
        "Gold",
        "Orchid",
        "LightSkyBlue"
    ],
    "bracket-pair-colorizer-2.forceIterationColorCycle": true,
    "trailing-spaces.trimOnSave": true,
    "trailing-spaces.backgroundColor": "rgba(160,160,160,0.3)",
    "trailing-spaces.borderColor": "rgba(160,160,160,0.3)",
    "trailing-spaces.highlightCurrentLine": false,
    "explorer.confirmDragAndDrop": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.vscode": true,
        "**/.DS_Store": true
    },
    "window.zoomLevel": 2,
    "tabnine.experimentalAutoImports": true,
    // python
    "python.pythonPath": "/anaconda3/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--ignore=E402"
    ],
    "python.linting.enabled": true,
    "python.linting.pycodestyleEnabled": false,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--ignore=E501,W503,E402,E701"
    ],
    "python.sortImports.args": [
        "-m 7"
    ],
    "autoDocstring.docstringFormat": "numpy",
    "python.languageServer": "Microsoft",
    // markdown
    "markdown.preview.breaks": true,
    "markdown.marp.themes": [
        "./marp/custom.css"
    ],
    // html
    "[html]": {
        "editor.tabSize": 2
    }
}
```