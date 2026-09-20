// Gazebo Harmonic system: prescribed kinematics, observed post-step state.
// The CSV is deliberately written from ECM Pose, never from DesiredPose().
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>

namespace offline {
class MotionExport final : public gz::sim::System,
                           public gz::sim::ISystemConfigure,
                           public gz::sim::ISystemPreUpdate,
                           public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override {
    model_ = gz::sim::Model(entity);
    kind_ = sdf->Get<std::string>("kind");
    if (kind_ != "constant_velocity" && kind_ != "smooth_turn")
      throw std::runtime_error("unsupported motion kind");
    initial_ = sdf->Get<gz::math::Vector3d>("initial_position_m");
    velocity_ = sdf->Get<gz::math::Vector3d>("initial_velocity_mps");
    turnStart_ = sdf->Get<double>("turn_start_s");
    turnEnd_ = sdf->Get<double>("turn_end_s");
    turnAngle_ = sdf->Get<double>("turn_angle_rad");
    period_ = sdf->Get<double>("export_period_s");
    const auto path = sdf->Get<std::string>("output_csv");
    if (period_ <= 0 || turnEnd_ <= turnStart_ ||
        velocity_.Length() >= 343.0 || path.empty())
      throw std::runtime_error("invalid Gazebo motion/export configuration");
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_) throw std::runtime_error("cannot open Gazebo state CSV: " + path);
    output_ << "sim_time_s,x_m,y_m,z_m,qw,qx,qy,qz\n" << std::setprecision(17);
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    const double t = std::chrono::duration<double>(info.simTime).count();
    const auto position = DesiredPosition(t);
    const auto direction = DesiredVelocity(t);
    const double yaw = std::atan2(direction.Y(), direction.X());
    model_.SetWorldPoseCmd(ecm, gz::math::Pose3d(
        position.X(), position.Y(), position.Z(), 0, 0, yaw));
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    const double t = std::chrono::duration<double>(info.simTime).count();
    if (t <= lastTime_) throw std::runtime_error("Gazebo simulation time reset or duplicate");
    lastTime_ = t;
    if (t + 1e-10 < static_cast<double>(sampleIndex_) * period_) return;
    // A model is a direct child of the world. worldPose reads ECM state after
    // physics applies the pose command; commanded trajectory is never exported.
    const auto pose = gz::sim::worldPose(model_.Entity(), ecm);
    const auto &p = pose.Pos();
    const auto &q = pose.Rot();
    output_ << t << ',' << p.X() << ',' << p.Y() << ',' << p.Z() << ','
            << q.W() << ',' << q.X() << ',' << q.Y() << ',' << q.Z() << '\n';
    if (!output_) throw std::runtime_error("failed writing Gazebo state CSV");
    ++sampleIndex_;
  }

 private:
  gz::math::Vector3d DesiredVelocity(double t) const {
    if (kind_ == "constant_velocity") return velocity_;
    const double u = std::clamp((t - turnStart_) / (turnEnd_ - turnStart_), 0.0, 1.0);
    const double angle = turnAngle_ * (3*u*u - 2*u*u*u);
    const double c = std::cos(angle), s = std::sin(angle);
    return {c*velocity_.X()-s*velocity_.Y(),
            s*velocity_.X()+c*velocity_.Y(), velocity_.Z()};
  }

  gz::math::Vector3d DesiredPosition(double t) const {
    if (kind_ == "constant_velocity" || t <= turnStart_)
      return initial_ + velocity_ * t;
    const double active = std::min(t, turnEnd_) - turnStart_;
    // Composite Simpson integral of the prescribed velocity. Its integration
    // error is far below the 1 ms simulation / 20 ms export discretization.
    constexpr int n = 64;
    gz::math::Vector3d integral{0, 0, 0};
    for (int i = 0; i <= n; ++i) {
      const double weight = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
      integral += DesiredVelocity(turnStart_ + active * i / n) * weight;
    }
    integral *= active / (3*n);
    if (t > turnEnd_) integral += DesiredVelocity(turnEnd_) * (t - turnEnd_);
    return initial_ + velocity_ * turnStart_ + integral;
  }

  gz::sim::Model model_{gz::sim::kNullEntity};
  std::string kind_;
  gz::math::Vector3d initial_, velocity_;
  double turnStart_{}, turnEnd_{}, turnAngle_{}, period_{};
  double lastTime_{-1.0};
  unsigned long sampleIndex_{0};
  std::ofstream output_;
};
}  // namespace offline

GZ_ADD_PLUGIN(offline::MotionExport, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate,
              gz::sim::ISystemPostUpdate)
