#include "suction_gripper_hardware/suction_gripper_system.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>

#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace suction_gripper_hardware
{

SuctionGripperSystem::CallbackReturn SuctionGripperSystem::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  if (info_.gpios.size() != 1) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "Expected exactly one <gpio> entry.");
    return CallbackReturn::ERROR;
  }

  gpio_name_ = info_.gpios[0].name;

  if (info_.gpios[0].command_interfaces.size() != 1) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "Expected exactly one GPIO command interface.");
    return CallbackReturn::ERROR;
  }

  if (info_.gpios[0].state_interfaces.size() != 1) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "Expected exactly one GPIO state interface.");
    return CallbackReturn::ERROR;
  }

  interface_name_ = info_.gpios[0].command_interfaces[0].name;

  if (info_.hardware_parameters.count("device")) {
    device_ = info_.hardware_parameters.at("device");
  }

  if (info_.hardware_parameters.count("baud_rate")) {
    baud_rate_ = std::stoi(info_.hardware_parameters.at("baud_rate"));
  }

  const auto & state_interface = info_.gpios[0].state_interfaces[0];
  if (state_interface.parameters.count("initial_value")) {
    state_ = std::stod(state_interface.parameters.at("initial_value"));
    command_ = state_;
  }

  last_sent_logic_ = std::numeric_limits<double>::quiet_NaN();

  RCLCPP_INFO(
    rclcpp::get_logger("SuctionGripperSystem"),
    "Initialized suction gripper GPIO '%s/%s' on %s at %d baud",
    gpio_name_.c_str(),
    interface_name_.c_str(),
    device_.c_str(),
    baud_rate_);

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
SuctionGripperSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(
      gpio_name_,
      interface_name_,
      &state_));

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
SuctionGripperSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  command_interfaces.emplace_back(
    hardware_interface::CommandInterface(
      gpio_name_,
      interface_name_,
      &command_));

  return command_interfaces;
}

SuctionGripperSystem::CallbackReturn SuctionGripperSystem::on_configure(
  const rclcpp_lifecycle::State &)
{
  if (!open_uart()) {
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

SuctionGripperSystem::CallbackReturn SuctionGripperSystem::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  close_uart();
  return CallbackReturn::SUCCESS;
}

SuctionGripperSystem::CallbackReturn SuctionGripperSystem::on_activate(
  const rclcpp_lifecycle::State &)
{
  last_sent_logic_ = std::numeric_limits<double>::quiet_NaN();

  // Push current command once when activated.
  return write(rclcpp::Time{}, rclcpp::Duration::from_seconds(0.0)) ==
           hardware_interface::return_type::OK
           ? CallbackReturn::SUCCESS
           : CallbackReturn::ERROR;
}

SuctionGripperSystem::CallbackReturn SuctionGripperSystem::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type SuctionGripperSystem::read(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  if (fd_ < 0) {
    return hardware_interface::return_type::ERROR;
  }

  parse_rx();
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type SuctionGripperSystem::write(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  if (fd_ < 0) {
    return hardware_interface::return_type::ERROR;
  }

  const double logic = command_ >= 0.5 ? 1.0 : 0.0;

  if (!std::isnan(last_sent_logic_) && logic == last_sent_logic_) {
    return hardware_interface::return_type::OK;
  }

  const std::string uart_cmd = logic > 0.5 ? "pneumo1\n" : "pneumo0\n";

  if (!send_line(uart_cmd)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "Failed to write UART command '%s'",
      uart_cmd.c_str());
    return hardware_interface::return_type::ERROR;
  }

  last_sent_logic_ = logic;

  // Optimistic state update. If Arduino echoes pneumo1/pneumo0, read() will confirm it.
  state_ = logic;

  return hardware_interface::return_type::OK;
}

bool SuctionGripperSystem::open_uart()
{
  close_uart();

  fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

  if (fd_ < 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "Cannot open UART device '%s': %s",
      device_.c_str(),
      std::strerror(errno));
    return false;
  }

  if (!configure_uart()) {
    close_uart();
    return false;
  }

  ::tcflush(fd_, TCIOFLUSH);

  return true;
}

void SuctionGripperSystem::close_uart()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SuctionGripperSystem::configure_uart()
{
  termios tty{};

  if (::tcgetattr(fd_, &tty) != 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "tcgetattr failed: %s",
      std::strerror(errno));
    return false;
  }

  const int baud_constant = baud_to_constant(baud_rate_);

  ::cfmakeraw(&tty);
  ::cfsetispeed(&tty, baud_constant);
  ::cfsetospeed(&tty, baud_constant);

  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSTOPB;

#ifdef CRTSCTS
  tty.c_cflag &= ~CRTSCTS;
#endif

  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SuctionGripperSystem"),
      "tcsetattr failed: %s",
      std::strerror(errno));
    return false;
  }

  return true;
}

bool SuctionGripperSystem::send_line(const std::string & line)
{
  const auto * data = line.data();
  size_t remaining = line.size();

  while (remaining > 0) {
    const ssize_t written = ::write(fd_, data, remaining);

    if (written < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return false;
    }

    remaining -= static_cast<size_t>(written);
    data += written;
  }

  ::tcdrain(fd_);
  return true;
}

void SuctionGripperSystem::parse_rx()
{
  char buffer[128];

  while (true) {
    const ssize_t n = ::read(fd_, buffer, sizeof(buffer));

    if (n > 0) {
      rx_buffer_.append(buffer, static_cast<size_t>(n));
    } else {
      if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        RCLCPP_WARN(
          rclcpp::get_logger("SuctionGripperSystem"),
          "UART read failed: %s",
          std::strerror(errno));
      }
      break;
    }
  }

  size_t pos = 0;
  while ((pos = rx_buffer_.find('\n')) != std::string::npos) {
    std::string line = rx_buffer_.substr(0, pos);
    rx_buffer_.erase(0, pos + 1);

    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }

    if (line == "pneumo1") {
      state_ = 1.0;
    } else if (line == "pneumo0") {
      state_ = 0.0;
    }
  }
}

int SuctionGripperSystem::baud_to_constant(int baud)
{
  switch (baud) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    default:
      throw std::runtime_error("Unsupported baud rate: " + std::to_string(baud));
  }
}

}  // namespace suction_gripper_hardware

PLUGINLIB_EXPORT_CLASS(
  suction_gripper_hardware::SuctionGripperSystem,
  hardware_interface::SystemInterface)