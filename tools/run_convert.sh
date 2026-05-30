#!/bin/bash

HOST="192.168.3.231"
USER_NAME="root"
PASS="TDghmtonolSq8uLb"

export HOST USER_NAME PASS

expect <<'EOF'
set timeout -1
set HOST $env(HOST)
set USER_NAME $env(USER_NAME)
set PASS $env(PASS)

# Step 1: 上传转换脚本
puts "\n========== Step 1: 上传转换脚本 ==========\n"
spawn rsync -avz /Users/ck/Downloads/yolo11_backup/tools/yolo2coco.py $USER_NAME@$HOST:/tmp/yolo2coco.py
expect {
    "*yes/no*" { send "yes\r"; exp_continue }
    "*assword*" { send "$PASS\r"; exp_continue }
    eof
}

# Step 2: SSH 登录
puts "\n========== Step 2: SSH 登录并查看数据集结构 ==========\n"
spawn ssh $USER_NAME@$HOST
expect {
    "*yes/no*" { send "yes\r"; exp_continue }
    "*assword*" { send "$PASS\r"; exp_continue }
    -re {[$#>%]\s*} {}
}

# 先看看数据集结构
send "ls -la /home/dataList/drone/drone_car/\r"
expect -re {.*[$#>%]\s*}

send "ls -la /home/dataList/drone/drone_car/images/ 2>/dev/null | head -5\r"
expect -re {.*[$#>%]\s*}

send "ls -la /home/dataList/drone/drone_car/labels/ 2>/dev/null | head -5\r"
expect -re {.*[$#>%]\s*}

send "cat /home/dataList/drone/drone_car/data.yaml 2>/dev/null || cat /home/dataList/drone/drone_car/dataset.yaml 2>/dev/null || echo 'NO YAML FOUND'\r"
expect -re {.*[$#>%]\s*}

# 看看标签文件示例
send "find /home/dataList/drone/drone_car -name '*.txt' -path '*/labels/*' | head -3\r"
expect -re {.*[$#>%]\s*}

send "head -5 \$(find /home/dataList/drone/drone_car -name '*.txt' -path '*/labels/*' | head -1) 2>/dev/null\r"
expect -re {.*[$#>%]\s*}

# Step 3: 执行转换
puts "\n========== Step 3: 执行 YOLO → COCO 转换 ==========\n"
send "cd /tmp && python3 yolo2coco.py --src /home/dataList/drone/drone_car --dst /home/dataList/drone/drone_car_coco\r"
expect -re {.*[$#>%]\s*}

# 检查结果
send "ls -la /home/dataList/drone/drone_car_coco/\r"
expect -re {.*[$#>%]\s*}

send "ls -la /home/dataList/drone/drone_car_coco/annotations/ 2>/dev/null\r"
expect -re {.*[$#>%]\s*}

puts "\n========== 转换完成 ==========\n"
send "exit\r"
expect eof
EOF
