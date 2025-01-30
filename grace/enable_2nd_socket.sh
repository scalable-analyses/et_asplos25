for core in $(seq 72 143); do
    echo "enabling core $core"
    echo 1 > /sys/devices/system/cpu/cpu$core/online
done