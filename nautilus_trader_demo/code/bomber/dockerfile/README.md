# Bomber 远程开发与编译环境 (二进制交付 / 源码隔离版)

本目录提供了一套用于远程开发、回测与策略运行的 Docker 容器环境。

## 核心特性
1. **源码彻底隔离（纯二进制交付）**：
   - 采用多阶段构建（Multi-stage Build），仅在内部编译阶段使用源码生成 Wheel 二进制包；
   - 最终交付并运行的镜像中**不包含任何 Bomber 底层源码、Git 历史及 Rust 编译工具**，仅包含编译好的 Python 库文件（`site-packages`）。
2. **VS Code Remote-SSH 支持**：
   - 容器内常驻 OpenSSH Server，默认映射端口为 `2222`；
   - 支持非 root 用户（默认 `dev`）及安全的 SSH 公钥认证。
3. **独立的策略工作区**：
   - 默认将宿主机的 `./dockerfile/workspace` 挂载至容器内的 `/home/dev/workspace`，用户编写的交易策略和回测脚本持久化保存在此目录，不与 Bomber 核心代码混杂。

---

## 快速上手

### 1. 配置认证方式（支持密码登录与证书登录）

- **方式一：账号密码登录（默认开箱即用）**
  - 默认用户名：`dev`
  - 默认密码：`bomber123`（可在 `compose.remote.yml` 或启动命令中通过 `DEV_PASSWORD` 修改）

- **方式二：证书公钥免密登录（可选）**
  - 在 `dockerfile/` 目录下放置你的公钥：
    ```bash
    cat ~/.ssh/id_ed25519.pub > dockerfile/authorized_keys
    chmod 600 dockerfile/authorized_keys
    ```
  - 取消 `dockerfile/compose.remote.yml` 中 `secrets` 的相关注释即可。

---

### 2. 构建镜像并启动容器

在仓库根目录下执行：

```bash
# 1. 创建策略工作区目录（若不存在）
mkdir -p dockerfile/workspace

# 2. 构建镜像（如果此前已构建，这一步会秒级完成）
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f dockerfile/compose.remote.yml build

# 3. 后台启动容器（支持通过环境变量指定数据目录、端口、密码）
# 基本启动（数据目录默认为 ./data，密码默认为 bomber123，端口默认为 2222）
docker compose -f dockerfile/compose.remote.yml up -d

# 自定义宿主机数据目录启动（例如服务器数据存放于 /worker1/data）：
DATA_DIR=/worker1/data docker compose -f dockerfile/compose.remote.yml up -d

# 组合参数启动示例（指定数据目录 + 外部端口 + 登录密码）：
DATA_DIR=/worker1/data SSH_PORT=22222 DEV_PASSWORD="your_password" docker compose -f dockerfile/compose.remote.yml up -d

# 4. 查看启动状态与日志
docker compose -f dockerfile/compose.remote.yml ps
docker compose -f dockerfile/compose.remote.yml logs -f
```

---

### 3. SSH 连接与验证

#### 命令行直连
```bash
ssh -p 2222 dev@127.0.0.1
# 若位于远程服务器，则替换为远程服务器 IP:
# ssh -p 2222 dev@<服务器IP>
# 提示输入密码时输入: bomber123 (或自定义的 DEV_PASSWORD)
```

#### 登录后环境验证
进入容器后，默认处于 `/home/dev/workspace` 目录：
```bash
# 验证 Python 版本
python --version

# 验证 Bomber 二进制库已就绪（无源码，直接导入）
python -c "import bomber; print('Bomber version:', bomber.__version__)"

# 验证数据目录环境变量与挂载
echo "数据目录路径: $BOMBER_DATA_DIR"
ls -la "$BOMBER_DATA_DIR"

# 验证没有底层源码目录
ls /workspace  # 提示 No such file or directory
```

---

### 4. VS Code Remote-SSH 连接配置

在本地电脑的 `~/.ssh/config` 中添加如下配置：

```ssh-config
Host bomber-remote
    HostName <服务器IP或127.0.0.1>
    Port 2222
    User dev
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

然后在 VS Code 中按 `F1` 或 `Ctrl+Shift+P` (Mac 为 `Cmd+Shift+P`)：
1. 选择 `Remote-SSH: Connect to Host...` -> 点击 `bomber-remote`。
2. 连接成功后，打开目录：`/home/dev/workspace`。
3. Python 解释器直接选择系统默认的 `/usr/local/bin/python` 即可。

---

### 5. 停止与清理

```bash
# 停止容器
docker compose -f dockerfile/compose.remote.yml stop

# 销毁容器（工作区策略代码保留在宿主机 dockerfile/workspace）
docker compose -f dockerfile/compose.remote.yml down
```
