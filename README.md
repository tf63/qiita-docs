# Qiita の執筆環境

### Qiita CLIのセットアップ
https://github.com/increments/qiita-cli

```
    # インストール
    npm install @qiita/qiita-cli --save-dev
    # インストール確認
    npx qiita version
    # 最近バージョンにアップデート
    npm install @qiita/qiita-cli@latest
    # 初期化
    npx qiita init
```

### Qiita CLIの使い方
```
    🚀 コンテンツをブラウザでプレビューする
    npx qiita preview

    🚀 新しい記事を追加する
    npx qiita new (記事のファイルのベース名)

    🚀 記事を投稿、更新する
    npx qiita publish (記事のファイルのベース名)

    💁 コマンドのヘルプを確認する
    npx qiita help
```
