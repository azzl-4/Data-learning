# main.spec 把Python應用打包成一個可執行的檔案。成獨立的應用程序
# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# 設置模型文件的絕對路徑
base_path = 'D:\\topic2\\Data-learning\\'

a = Analysis(
    ['main.py'],
    pathex=['.'],  # 設定當前工作目錄
    binaries=[],
    datas=[
        (os.path.join(base_path, 'best_knn_model.pkl'), '.'), 
        (os.path.join(base_path, 'best_svm_model.pkl'), '.'), 
        (os.path.join(base_path, 'svm_model.joblib'), '.'), 
        (os.path.join(base_path, 'logistic_regression_model.joblib'), '.'), 
        (os.path.join(base_path, 'word2vec_model.pkl'), '.'), 
        (os.path.join(base_path, 'tfidf_dict.pkl'), '.'), 
        (os.path.join(base_path, 'vectorizer.joblib'), '.'), 
        (os.path.join(base_path, 'suicide_detection_roberta_model'), 'suicide_detection_roberta_model')
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SuicideRiskApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SuicideRiskApp'
)
