#ifndef SUCTION_GRIPPER_HARDWARE__SUCTION_GRIPPER_SYSTEM_HPP_
#define SUCTION_GRIPPER_HARDWARE__SUCTION_GRIPPER_SYSTEM_HPP_

#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace suction_gripper_hardware
{

class SuctionGripperSystem : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(SuctionGripperSystem)

  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface>
  export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface>
  export_command_interfaces() override;

  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  bool open_uart();
  void close_uart();
  bool configure_uart();
  bool send_line(const std::string & line);
  void parse_rx();
  static int baud_to_constant(int baud);

  std::string device_ = "/dev/ttyACM0";
  int baud_rate_ = 115200;

  std::string gpio_name_ = "suction_gripper";
  std::string interface_name_ = "state";

  int fd_ = -1;

  double command_ = 0.0;
  double state_ = 0.0;
  double last_sent_logic_ = -1.0;

  std::string rx_buffer_;
};

}  // namespace suction_gripper_hardware

#endif  // SUCTION_GRIPPER_HARDWARE__SUCTION_GRIPPER_SYSTEM_HPP_