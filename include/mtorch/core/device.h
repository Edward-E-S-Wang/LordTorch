#pragma once

namespace mtorch {

enum class DeviceType {
    CPU
};

struct Device {
    DeviceType type = DeviceType::CPU;
};

} // namespace mtorch
