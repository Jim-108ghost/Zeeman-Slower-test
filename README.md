# yb2-fluoresence

03/2026 塞满减速器数值模拟

03/16 test: 创建了新fork 设置了upstream
[人人都能看懂的Git教程！Git如何和 GitHub、GitLab 交互？团队如何用 Git 协作开发？小白也能看懂的Git教程！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1d6XVYqEuy/?spm_id_from=333.788.comment.all.click&vd_source=d14f545da19587e6aacc915f293d4eba)

## 1. fork 仓库

- 打开 [原代码仓库](https://github.com/Jim-108ghost/Zeeman-Slower-test)
- 点击右上角 **Fork** → 选你的账号

## 2. clone 到本地

- 在你电脑上找个地方（比如桌面），**右键 → Git Bash Here**
  
- 输入：`git clone https://github.com/<你的用户名>/my-forked-repo`

- 不一定非得叫my-forked-repo叫别的也行总之这是另一个仓库，会显示在你的profile里。

## 3. 设置upstream链接主仓库:

- 在git bash里打开你fork的本地仓库：`cd my-forked-repo`
- 设置上游：`git remote add upstream https://github.com/Jim-108ghost/Zeeman-Slower-test`

## 4. 日常工作流

- 同步一下主仓库的代码

```bash
# 获取主仓库最新代码
git fetch upstream
# 切换到主分支并合并
git checkout main
git merge upstream/main
# 把你的github仓库也更新一下
git push origin main
```

- 创建功能分支避免主仓库那边不merge你PR这边没法继续（我也不是很清楚先这么搞一下试试吧）
  
```bash
# 分支名要清晰，例如：
git checkout -b feature/加一个新功能
git checkout -b bugfix/修一个bug
```

## 5. 写完代码提交（commit）

```bash
# 查看改了哪些文件
git status
# 添加某一个改动
git add README.md
# 或者添加所有改动
git add .
# 提交到本地（写清楚改了啥）
git commit -m "完成了xxx功能"
```

## 6. 推到你自己的fork仓库
`git push origin feature/加一个新功能`

## 7. 发起PR：
打开你的 GitHub 仓库页面，会看到一个黄色的 Compare & pull request按钮。点击后填写说明，然后 Create pull request。在评论里 @Jim-108ghost 让他 review

## 8. 合并完清理分支

```bash
# 切回主分支
git checkout main
# 删除本地功能分支
git branch -d feature/加一个新功能
# 删除github上的分支
git push origin --delete feature/加一个新功能
```