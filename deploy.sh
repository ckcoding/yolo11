#!/bin/bash

# ================= 配置区 =================
# 所有敏感信息均从环境变量读取，请勿在此文件中硬编码任何密码！
# ==========================================

# 从环境变量读取敏感信息（请勿硬编码密码！）
# 使用方式: DEPLOY_HOST=x DEPLOY_USER=x DEPLOY_PASS=x bash deploy.sh
HOST="${DEPLOY_HOST:?请设置 DEPLOY_HOST 环境变量}"
USER_NAME="${DEPLOY_USER:?请设置 DEPLOY_USER 环境变量}"
PASS="${DEPLOY_PASS:?请设置 DEPLOY_PASS 环境变量}"
TARGET_DIR="${DEPLOY_TARGET_DIR:-/home/ck/yolo11_backup}"
# 获取当前工作目录
LOCAL_DIR="$(pwd)/"

export HOST USER_NAME PASS TARGET_DIR LOCAL_DIR

expect <<'EOF'
# 无超时限制，以防止传输中途断开
set timeout -1

set HOST $env(HOST)
set USER_NAME $env(USER_NAME)
set PASS $env(PASS)
set TARGET_DIR $env(TARGET_DIR)
set LOCAL_DIR $env(LOCAL_DIR)

puts "\n========== Step 1: 同步代码到远程主机 ==========\n"
# rsync 需要将带空格的参数正确用双引号包裹为一整个参数
spawn rsync -avz --exclude=".git" --exclude=".DS_Store" --exclude="__pycache__" --exclude="node_modules" --exclude="models" "--rsync-path=sudo rsync" $LOCAL_DIR $USER_NAME@$HOST:$TARGET_DIR

# 捕获 SSH 密钥指纹和各种密码请求
expect {
    "*yes/no*" { send "yes\r"; exp_continue }
    "*assword*" { send "$PASS\r"; exp_continue }
    eof
}

puts "\n========== Step 2: SSH 登录并执行命令 ==========\n"
spawn ssh $USER_NAME@$HOST

expect {
    "*yes/no*" { send "yes\r"; exp_continue }
    "*assword*" { 
        send "$PASS\r"
        exp_continue 
    }
    -re {[$#>%]\s*} {
        # 匹配到终端提示符，说明登录成功
    }
}

# 等待普通用户的命令提示符并强制切换为 root (执行 sudo su -)
send "sudo su -\r"

# 处理 sudo su - 的密码提示（如果已经免密会直接到 root）
expect {
    "*assword*" { 
        send "$PASS\r"
        expect -re {.*#\s*}
    }
    -re {.*#\s*} {
        # 拿到 root 权限会包含 '#' 字符
    }
}

puts "\n========== Step 3: 开始进行相关构建与重启 ==========\n"

# --------- 以下步骤都在 root 权限下进行 ---------
send "cd $TARGET_DIR/stream_service\r"
expect -re {.*#\s*}

# 按当前 docker-compose 的信息推断，重启 stream 服务
send "docker-compose down\r"
expect -re {.*#\s*}

send "docker-compose up -d --build\r"
expect -re {.*#\s*}
# --------------------------------------------------

puts "\n========== 部署运行完成，开始退出并断开 ==========\n"

# 退出 root登录环境
send "exit\r"

# 退出 普通用户 的远程 ssh
expect -re {[$#>%]\s*}
send "exit\r"

expect eof
EOF

