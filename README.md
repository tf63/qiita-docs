# Qiita の執筆環境

### Qiita CLIのセットアップ
https://github.com/increments/qiita-cli

```
    # インストール
    pnpm install @qiita/qiita-cli --save-dev
    # インストール確認
    pnpm exec qiita version
    # 最近バージョンにアップデート
    pnpm install @qiita/qiita-cli@latest
    # 初期化
    pnpm exec qiita init
```

### Qiita CLIの使い方
```
    🚀 コンテンツをブラウザでプレビューする
    pnpm exec qiita preview

    🚀 新しい記事を追加する
    pnpm exec qiita new (記事のファイルのベース名)

    🚀 記事を投稿、更新する
    pnpm exec qiita publish (記事のファイルのベース名)

    💁 コマンドのヘルプを確認する
    pnpm exec qiita help
```
