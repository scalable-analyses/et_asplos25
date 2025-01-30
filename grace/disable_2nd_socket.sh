for core in $(seq 72 143); do
    echo "disabling core $core"
    echo 0 > /sys/devices/system/cpu/cpu$core/online
done