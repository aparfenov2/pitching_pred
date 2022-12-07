export PATH=~/.local/bin:$PATH
[ "$1" == "--saved" ] && {
    tensorboard --logdir=./saved/
    exit 0
}
tensorboard --logdir=./lightning_logs/