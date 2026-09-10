#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${REMOTE_USER:-dev}"
USER_HOME="$(getent passwd "${USER_NAME}" | cut -d: -f6)"
KEY_FILE="/run/secrets/dev_authorized_key"

USER_PASSWORD="${DEV_PASSWORD:-bomber123}"

# 1. 设置用户密码（支持密码登录）
echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd
echo "Password configured for user '${USER_NAME}'."

mkdir -p "${USER_HOME}/.ssh"

# 2. 配置 SSH 公钥（可选，若提供则生效）
if [ -s "${KEY_FILE}" ]; then
    install \
        -o "${USER_NAME}" \
        -g "${USER_NAME}" \
        -m 0600 \
        "${KEY_FILE}" \
        "${USER_HOME}/.ssh/authorized_keys"
    echo "SSH authorized_keys configured from Docker Secret."
elif [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    printf '%s\n' "${SSH_PUBLIC_KEY}" > "${USER_HOME}/.ssh/authorized_keys"
    chown "${USER_NAME}:${USER_NAME}" "${USER_HOME}/.ssh/authorized_keys"
    chmod 0600 "${USER_HOME}/.ssh/authorized_keys"
    echo "SSH authorized_keys configured from SSH_PUBLIC_KEY environment variable."
else
    echo "Notice: No SSH public key provided, password authentication enabled."
fi

chown -R "${USER_NAME}:${USER_NAME}" "${USER_HOME}/.ssh"
chmod 0700 "${USER_HOME}/.ssh"

# 3. 保证工作区和数据目录存在且属于 dev 用户
mkdir -p "${USER_HOME}/workspace"
chown "${USER_NAME}:${USER_NAME}" "${USER_HOME}/workspace"

DATA_PATH="${BOMBER_DATA_DIR:-${USER_HOME}/data}"
mkdir -p "${DATA_PATH}"
chown "${USER_NAME}:${USER_NAME}" "${DATA_PATH}" 2>/dev/null || true

# 4. 同步运行时数据目录环境变量至 SSH 会话
if [ -n "${BOMBER_DATA_DIR:-}" ]; then
    echo "export BOMBER_DATA_DIR=\"${BOMBER_DATA_DIR}\"" >> /etc/profile.d/00-env.sh
    sed -i '/^BOMBER_DATA_DIR=/d' /etc/environment 2>/dev/null || true
    echo "BOMBER_DATA_DIR=\"${BOMBER_DATA_DIR}\"" >> /etc/environment
fi
if [ -n "${BT_DATA_DIR:-}" ]; then
    echo "export BT_DATA_DIR=\"${BT_DATA_DIR}\"" >> /etc/profile.d/00-env.sh
    sed -i '/^BT_DATA_DIR=/d' /etc/environment 2>/dev/null || true
    echo "BT_DATA_DIR=\"${BT_DATA_DIR}\"" >> /etc/environment
fi

echo "Starting OpenSSH Daemon..."
exec /usr/sbin/sshd -D -e
