---
title: Huggingfaceのdiffusersでpretrainedのモデルの保存先を変更する
tags:
  - Python
  - PyTorch
  - huggingface
  - StableDiffusion
private: false
updated_at: '2024-01-09T11:46:21+09:00'
id: 543decd46bbab9de7aba
organization_url_name: null
slide: false
ignorePublish: false
---
Huggingfaceの[diffusers](https://huggingface.co/docs/diffusers/index)では`pipeline.from_pretrained`を実行することでHuggingface Hubからpretainedのモデルをダウンロードできます

ダウンロードしたモデルは`~/.cache/huggingface`にキャッシュとして保存されます

Docker環境ではキャッシュ先のディレクトリをマウントしておかないとコンテナを破棄したタイミングでキャッシュが失われてしまいます

今回はキャッシュ先のディレクトリを`~/.cache/huggingface`から`<work_dir>/.cache/huggingface`に変更してみます

試してみたところ，キャッシュ先のディレクトリは環境変数`$HF_HUB_CACHE`で指定されています．`$HF_HUB_CACHE`は`$HF_HOME/hub`に設定されているので，環境変数`$HF_HOME`を変えれば良さそうです

```bash
    export HF_HOME=<work_dir>/.cache/huggingface
```

https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache

Docker環境ではENVコマンドで環境変数を設定できます
```Dockerfile
    FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
    LABEL maintainer="Hugging Face"
    LABEL repository="diffusers"
    
    # ワークディレクトリの指定
    WORKDIR /app

    # 非インタラクティブモードにする (入力待ちでブロックしなくなる)
    ENV DEBIAN_FRONTEND noninteractive
    # .pycを作らないように
    ENV PYTHONDONTWRITEBYTECODE 1
    # バッファの無効化
    ENV PYTHONUNBUFFERED 1
    # torchvisionでpretrainedのモデルを保存する場所
    ENV TORCH_HOME /app/.cache/torchvision
    
    # ----------------------------------------------------------------
    # setup (root) 
    # ----------------------------------------------------------------
    RUN apt update && \
        apt install -y bash \
        build-essential \
        git \
        git-lfs \
        curl \
        ca-certificates \
        libsndfile1-dev \
        libgl1 \
        python3.8 \
        python3-pip \
        python3.8-venv && \
        rm -rf /var/lib/apt/lists
    
    # ----------------------------------------------------------------
    # create user
    # ----------------------------------------------------------------
    # UIDとGIDは外から与える
    ARG USER_UID
    ARG USER_GID
    
    # コンテナ内でのユーザー名， グループ名
    ARG USER_NAME=user
    ARG GROUP_NAME=user
    
    # グループが存在しなかったら，　適当なグループを作成
    RUN if ! getent group $USER_GID >/dev/null; then \
        groupadd -g $USER_GID $GROUP_NAME; \
        fi
    
    # ユーザーを作成
    RUN useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USER_NAME
    
    # 初期ユーザーの変更
    USER $USER_NAME
    
    # ----------------------------------------------------------------
    # setup (user) 
    # ----------------------------------------------------------------
    
    # pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
    RUN python3 -m pip install --no-cache-dir --upgrade pip && \
        python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        invisible_watermark && \
        python3 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy \
        scipy \
        tensorboard \
        transformers \
        omegaconf \
        pytorch-lightning \
        xformers
    
    ENV HF_HOME /app/.cache/huggingface
    
    CMD ["/bin/bash"]
```

終わり
