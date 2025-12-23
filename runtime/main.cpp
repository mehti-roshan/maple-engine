#include <engine/core/engine.h>

int main(int argc, char** argv) {
  maple::Engine app{};
  app.Init();
  app.Run();
  return 0;
}