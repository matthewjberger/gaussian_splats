mod gaussian;
mod ply;
mod splat_pass;

use gaussian::GpuGaussian;
use nightshade::prelude::*;
use splat_pass::SplatPass;

static GAUSSIANS: std::sync::OnceLock<Vec<GpuGaussian>> = std::sync::OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ply_path = std::env::args()
        .nth(1)
        .expect("Usage: gaussian_splats <path_to.ply>");

    let raw_gaussians = ply::load_ply(std::path::Path::new(&ply_path));
    let gpu_gaussians: Vec<GpuGaussian> = raw_gaussians.iter().map(GpuGaussian::from).collect();

    let gaussian_count = gpu_gaussians.len();
    eprintln!("Loaded {} gaussians from {}", gaussian_count, ply_path);

    if GAUSSIANS.set(gpu_gaussians).is_err() {
        panic!("Failed to set gaussians");
    }

    launch(GaussianSplatViewer { gaussian_count })?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();

    wasm_bindgen_futures::spawn_local(async {
        let bytes = fetch_ply_bytes("assets/truck_point_cloud.ply").await;

        let raw_gaussians = ply::load_ply_from_bytes(&bytes);
        let gpu_gaussians: Vec<GpuGaussian> = raw_gaussians.iter().map(GpuGaussian::from).collect();

        let gaussian_count = gpu_gaussians.len();

        if GAUSSIANS.set(gpu_gaussians).is_err() {
            panic!("Failed to set gaussians");
        }

        launch(GaussianSplatViewer { gaussian_count }).expect("Failed to launch");
    });
}

#[cfg(target_arch = "wasm32")]
async fn fetch_ply_bytes(url: &str) -> Vec<u8> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().expect("No window");
    let response = JsFuture::from(window.fetch_with_str(url))
        .await
        .expect("Fetch failed");
    let response: web_sys::Response = response.dyn_into().expect("Response cast failed");
    let array_buffer = JsFuture::from(response.array_buffer().expect("Array buffer failed"))
        .await
        .expect("Array buffer await failed");
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    uint8_array.to_vec()
}

struct GaussianSplatViewer {
    gaussian_count: usize,
}

impl State for GaussianSplatViewer {
    fn title(&self) -> &str {
        "3D Gaussian Splatting Viewer"
    }

    fn initialize(&mut self, world: &mut World) {
        world.resources.user_interface.enabled = true;
        world.resources.graphics.show_grid = false;
        world.resources.graphics.atmosphere = Atmosphere::None;

        let camera_entity = spawn_pan_orbit_camera(
            world,
            Vec3::new(0.0, 0.0, 0.0),
            5.0,
            0.0,
            std::f32::consts::FRAC_PI_4,
            "Main Camera".to_string(),
        );
        world.resources.active_camera = Some(camera_entity);
    }

    fn configure_render_graph(
        &mut self,
        graph: &mut RenderGraph<World>,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        resources: RenderResources,
    ) {
        let gaussians = GAUSSIANS.get().expect("Gaussians not loaded");

        let splat_pass = SplatPass::new(device, gaussians, wgpu::TextureFormat::Rgba16Float);

        graph
            .pass(Box::new(splat_pass))
            .slot("color", resources.scene_color)
            .slot("depth", resources.depth);

        let blit_pass = passes::BlitPass::new(device, surface_format);
        graph
            .pass(Box::new(blit_pass))
            .read("input", resources.scene_color)
            .write("output", resources.swapchain);
    }

    fn run_systems(&mut self, world: &mut World) {
        pan_orbit_camera_system(world);
    }

    fn ui(&mut self, world: &mut World, ui_context: &egui::Context) {
        egui::Window::new("Gaussian Splatting").show(ui_context, |ui| {
            ui.label(format!("Gaussians: {}", self.gaussian_count));

            let fps = 1.0 / world.resources.window.timing.delta_time.max(0.001);
            ui.label(format!("FPS: {:.1}", fps));
        });
    }

    fn on_keyboard_input(&mut self, world: &mut World, key_code: KeyCode, key_state: KeyState) {
        if matches!((key_code, key_state), (KeyCode::KeyQ, KeyState::Pressed)) {
            world.resources.window.should_exit = true;
        }
    }
}
