// Read-only Gazebo system: record the X500 base_link state after each physics step.
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>

namespace acoustic {
class X500Observer final : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate,
                          public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity &,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override {
    modelName_ = sdf->Get<std::string>("model_name");
    phasePath_ = sdf->Get<std::string>("phase_file");
    period_ = sdf->Get<double>("export_period_s");
    const auto outputPath = sdf->Get<std::string>("output_csv");
    if (modelName_.empty() || outputPath.empty() || phasePath_.empty() ||
        !std::isfinite(period_) || period_ <= 0.0)
      throw std::runtime_error("invalid X500 observer configuration");
    output_.open(outputPath, std::ios::out | std::ios::trunc);
    if (!output_) throw std::runtime_error("cannot open X500 state CSV");
    output_ << "sim_time_s,x_m,y_m,z_m,qw,qx,qy,qz,"
               "vx_mps,vy_mps,vz_mps,flight_phase\n" << std::setprecision(17);
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    if (linkEntity_ != gz::sim::kNullEntity) return;
    const auto entity = ecm.EntityByComponents(
        gz::sim::components::Model(), gz::sim::components::Name(modelName_));
    if (entity == gz::sim::kNullEntity) return;
    linkEntity_ = gz::sim::Model(entity).CanonicalLink(ecm);
    if (linkEntity_ == gz::sim::kNullEntity)
      throw std::runtime_error("X500 has no canonical link");
    gz::sim::Link(linkEntity_).EnableVelocityChecks(ecm);
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override {
    if (info.paused || linkEntity_ == gz::sim::kNullEntity) return;
    const double t = std::chrono::duration<double>(info.simTime).count();
    if (t <= lastTime_)
      throw std::runtime_error("Gazebo simulation time reset or duplicate");
    lastTime_ = t;
    if (nextSampleTime_ >= 0.0 && t + 1e-10 < nextSampleTime_) return;
    const gz::sim::Link link(linkEntity_);
    const auto velocity = link.WorldLinearVelocity(ecm);
    if (!velocity) return;  // Wait for the first physics update after checks are enabled.
    // worldPose reads the observed ECM state. This plugin never commands a pose.
    const auto pose = gz::sim::worldPose(linkEntity_, ecm);
    const auto &p = pose.Pos();
    const auto &q = pose.Rot();
    const std::string phase = ReadPhase();
    output_ << t << ',' << p.X() << ',' << p.Y() << ',' << p.Z() << ','
            << q.W() << ',' << q.X() << ',' << q.Y() << ',' << q.Z() << ','
            << velocity->X() << ',' << velocity->Y() << ',' << velocity->Z()
            << ',' << phase << '\n' << std::flush;
    if (!output_) throw std::runtime_error("failed writing X500 state CSV");
    nextSampleTime_ = t + period_;
  }

 private:
  std::string ReadPhase() const {
    std::ifstream input(phasePath_);
    std::string phase;
    if (!(input >> phase)) return "preflight";
    for (const char c : phase)
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
        throw std::runtime_error("invalid flight phase marker");
    return phase;
  }

  std::string modelName_, phasePath_;
  double period_{};
  double lastTime_{-1.0}, nextSampleTime_{-1.0};
  gz::sim::Entity linkEntity_{gz::sim::kNullEntity};
  std::ofstream output_;
};
}  // namespace acoustic

GZ_ADD_PLUGIN(acoustic::X500Observer, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate,
              gz::sim::ISystemPostUpdate)
