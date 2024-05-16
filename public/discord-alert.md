---
title: 10分で本日の Contribution 数を通知する Discord bot を導入する
tags:
  - ''
private: false
updated_at: ''
id: null
organization_url_name: null
slide: false
ignorePublish: false
---

今回は **GAS** x **Github GraphQL API** を使用して本日の Contribution 数を通知する Discord bot を作成してみます
通知だけなら **Webhook** を使用して無料かつサーバーレスに Discord bot を導入できます

**使用技術**
Google App Script (GAS), GitHub GraphQL API, Webhook

## Discord 上で Webhook を作成
**Webhook** はアプリケーション内で発生したイベントをトリガーとして他のアプリケーションへ通知を送る仕組みです．Discord が提供している **Webhook** を使用すれば，発行したURLにPOSTするだけで Discord のメッセージを送信できます
https://discord.com/developers/docs/resources/webhook

### Webhookの作成

Discord で適当なサーバーとチャンネルを作り，**テキストチャンネルの編集 > 連携サービス > 新しいウェブフック** から **Webhook** を作成できます

![](https://storage.googleapis.com/zenn-user-upload/9f0afdf09501-20240411.png)

適当な名前の **Webhook** が作成されるので編集して使います

![](https://storage.googleapis.com/zenn-user-upload/a45ed0017848-20240411.png)

**ウェブフックURLをコピー** から **Webhook** のURLを取得できます．このURLにリクエストボディを`{content: "message"}`のようにしてPOSTすればメッセージを送信できます

:::note info
リクエストボディに`username`とか`avatar_url`を設定すれば bot の名前とアバター画像を上書きできます
https://discord.com/developers/docs/resources/webhook#execute-webhook
:::

## GitHub GraphQL API を使用して本日の Contribution 数を取得する
GitHubはユーザー情報の取得やリポジトリの操作のためのAPIとして **GitHub REST API** と **GitHub GraphQL API** を公開しています
https://docs.github.com/ja/graphql

### GitHub GraphQL APIを叩いてみる

Github CLI から **GitHub GraphQL API** のお試しが可能です．今回必要となるデータは [これ](https://docs.github.com/en/graphql/reference/objects#contributioncalendar) です．`login: <user_name>`を自分のものに書き換えて試してみてください

:::note info
Github CLIをインストールしていない方はこちらでも試せるらしいです
https://docs.github.com/ja/graphql/overview/explorer
:::

```bash
    gh api graphql -f query='
                        query contributions {
                          user(login: <user_name>) {
                            contributionsCollection(to: "2024-04-10T00:00:00", from: "2024-04-09T00:00:00") {
                              contributionCalendar {
                                weeks {
                                  contributionDays {
                                    date
                                    contributionCount
                                  }
                                }
                              }
                            }
                          }
                        }
                        '
```
レスポンス
```bash
(略)
              "contributionDays": [
                {
                  "date": "2024-04-09",
                  "contributionCount": 2
                },
                {
                  "date": "2024-04-10",
                  "contributionCount": 1
                }
              ]
(略)
```

正しく Contribution 数を取得できているみたいです

:::note warn
  **Github REST API** でも試してみましたが，IssueやPRの作成も含めた Contribution 数の取得は難しそうでした
:::

## GAS から Github GraphQL API を叩く

**Google App Script (GAS)** は Google が提供するローコードプラットフォームです．Googleドライブ上でファイルを作成するだけで簡単に実行可能です．
Google サービスとの連携が主な用途ですが，定期実行スクリプトのデプロイのしやすさから **Webhook** の連携にも適していると思います

### Github API アクセストークンの取得
GitHub CLI 以外から **GitHub GraphQL API** にアクセスするにはトークンが必要なので取得します

GitHub の **Settings > Developer Settings > Tokens (classic)** から **Generate new token > Generate new token (classic)** を選択します

Expiration (トークンの有効期限) を適当に設定し，**user > read:user** の項目のみチェックを入れてトークンを生成します

![](https://storage.googleapis.com/zenn-user-upload/1facff694124-20240411.png)

`ghp_<...>` みたいな文字列がトークンです．**絶対に公開しないでください**

### GASでコーディング
**GAS** でコーディングします．JavaScript をベースとしたプログラミング言語なので，慣れている人は読みやすいと思います

```js
// パラメータ
const USERNAME = "<GitHubのユーザー名>"
const GITHUB_TOKEN = "<作成したGitHub APIのアクセストークン>"
const DISCORD_URL = "<発行したDiscord WebhookのURL>"
const GITHUB_URL = "https://api.github.com/graphql"

let today = new Date(new Date().toLocaleString("ja-JP", {timeZone: "Asia/Tokyo"}))

// GraphQLのクエリ
// 今日のContribution数を取得する
const query = `query contributions {
                          user(login: "${USERNAME}") {
                            contributionsCollection(to: "${today.toISOString()}", from: "${today.toISOString()}") {
                              contributionCalendar {
                                weeks {
                                  contributionDays {
                                    date
                                    contributionCount
                                  }
                                }
                              }
                            }
                          }
                        }
`

// Github APIからContribution数を取得する
function getNumOfContributions () {

  // リクエストのオプション
  let options = {
    "method": "GET",
    "headers": {
      "Authorization": `Bearer ${GITHUB_TOKEN}`,
      "Content-Type": "application/json"
    },
    "payload": JSON.stringify({ query })
  }

  // Github APIからデータを取得する
  let response = UrlFetchApp.fetch(GITHUB_URL, options)

  // Contribution数を取得する
  if (response.getResponseCode() === 200) {
    // 正しくレスポンスが返ってきた場合

    // レスポンスをパース
    let datas = JSON.parse(response.getContentText())

    // 適当にcontribution数を取り出す
    let contribution = datas.data.user.contributionsCollection.contributionCalendar.weeks[0].contributionDays[0].contributionCount

    return contribution
  } else {
    // レスポンスが返ってこなかった場合，エラーを投げる
    throw new Error("Github APIにアクセスできませんでした．")
  }
}

// DiscordのWebhookにメッセージを登録
function postMessage (message) {
  
  // 登録するメッセージ
  let payload = {
    "content": message
  }

  // リクエストのオプション
  let options = {
    "method": "POST",
    "payload": payload
  }

  // Webhookにリクエストを投げる
  UrlFetchApp.fetch(DISCORD_URL, options)
}

// 日付を文字列に変換する関数
function formatDate(date) {
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  
  return `${month}月${day}日 ${hours}時${minutes}分` 
}

// エントリポイント
function main () {

  try {
    // 今日のContribution数を取得してメッセージを送信する
    let contribution = getNumOfContributions() 

    postMessage(`【${formatDate(today)}】 ----   ${contribution} Contributions`)
  } catch (e) {
    // エラーが生じた場合，その内容を送信する
    console.error(e)
    postMessage(`【エラーが発生しました】 ${e}`)
  }
}
```

### 動作確認
上部タブから `main` 関数を指定し，実行をクリックします

:::note warn
  ｢デプロイ｣というボタンもありますが，クリックしなくて大丈夫です
:::

![gas-execute](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3291419/2440a4a9-0887-495c-ef4b-b3c3b1a44d65.png)

正しく動作すれば Discord bot からメッセージが送信されます

![](https://storage.googleapis.com/zenn-user-upload/3b84d8c8b289-20240411.png)

## トリガーを設定
最後に作成した `main` 関数を定時実行するように設定します

**GAS** の`トリガー`タブから **トリガーを追加** を選択します

![](https://storage.googleapis.com/zenn-user-upload/0cad1860c75d-20240411.png)

午後10時 ~ 11時のどこかのタイミングで実行されるように設定できます

![](https://storage.googleapis.com/zenn-user-upload/3e46c419f8e1-20240411.png)

これ以上詳細に時間を設定できませんが，十分に動作するかと思います


---

以上です
