use std::{cmp::{max, min}, mem::swap};

// use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Rad, Vector2, Vector3, Vector4, Zero};
use minifb::{Key, Window, WindowOptions};
use glam::{Mat3, Mat4, USizeVec3, Vec3, Vec4, Vec2};

struct Model {
    triangles: Vec<TriangleInModel>,
    translate: Vec3,
    rotate: Mat3,
    scale: f32
}

impl Model {
    fn transform(&self) -> Mat4 {
        let translate = Mat4::from_translation(self.translate);
        let rotate = Mat4::from_mat3(self.rotate);
        let scale = Mat4::from_scale(Vec3::splat(self.scale));
        translate * rotate * scale
    }

    fn transform_triangle(&self, triangle: &TriangleInModel) -> TriangleInModel {
        let transform = self.transform();
        let p1 = transform * Vec4::new(triangle.p1.x, triangle.p1.y, triangle.p1.z, 1.0);
        let p2 = transform * Vec4::new(triangle.p2.x, triangle.p2.y, triangle.p2.z, 1.0);
        let p3 = transform * Vec4::new(triangle.p3.x, triangle.p3.y, triangle.p3.z, 1.0);

        let p1 = Vec3::new(p1.x / p1.w, p1.y / p1.w, p1.z / p1.w);
        let p2 = Vec3::new(p2.x / p2.w, p2.y / p2.w, p2.z / p2.w);
        let p3 = Vec3::new(p3.x / p3.w, p3.y / p3.w, p3.z / p3.w);

        TriangleInModel { p1, p2, p3 }
    }

    fn transform_triangles_iter(&self) -> impl Iterator<Item = TriangleInModel> + '_ {
        self.triangles.iter().map(move |triangle| self.transform_triangle(triangle))
    }
}

fn parser_file(file_path: &str) -> (Vec<Vec3>, Vec<USizeVec3>) {
    let file = std::fs::read_to_string(file_path).unwrap();

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let lines = file.split("\n");
    let mut phase = 0;

    let mut vertices_count = 0;
    let mut triangles_count = 0;

    for line in lines {
        let mut parts = line.split(" ");
        if phase == 0 {
            if let Some(first) = parts.next() {
                if first == "end_header" {
                    phase = 1;
                } else if first == "element" {
                    let element = parts.next().unwrap();
                    let count = parts.next().unwrap().parse::<usize>().unwrap();
                    if element == "vertex" {
                        vertices_count = count;
                    } else if element == "face" {
                        triangles_count = count;
                    }
                }
            }
        } else if phase == 1 {
            let mut x: Option<f32> = None;
            let mut y: Option<f32> = None;
            let mut z: Option<f32> = None;
            for part in parts {
                if x.is_none() {
                    x = Some(part.parse::<f32>().unwrap());
                } else if y.is_none() {
                    y = Some(part.parse::<f32>().unwrap());
                } else if z.is_none() {
                    z = Some(part.parse::<f32>().unwrap());
                }
            }
            vertices.push(Vec3::new(x.unwrap(), y.unwrap(), z.unwrap()));
            if vertices.len() == vertices_count {
                phase = 2;
            }
        } else if phase == 2 {
            let mut x: Option<usize> = None;
            let mut y: Option<usize> = None;
            let mut z: Option<usize> = None;

            _ = parts.next();
            
            for part in parts {
                if x.is_none() {
                    x = Some(part.parse::<usize>().unwrap());
                } else if y.is_none() {
                    y = Some(part.parse::<usize>().unwrap());
                } else if z.is_none() {
                    z = Some(part.parse::<usize>().unwrap());
                }
            }
            triangles.push(USizeVec3::new(x.unwrap(), y.unwrap(), z.unwrap()));
            if triangles.len() == triangles_count {
                break;
            }
        }
    }
    return (vertices, triangles);
}

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn main() {
    let (vertices, faces) = parser_file("/Users/mcfans/Downloads/bunny/reconstruction/bun_zipper.ply");

    let triangles: Vec<TriangleInModel> = faces.iter().map(|face| {
        let p1 = vertices[face.x];
        let p2 = vertices[face.y];
        let p3 = vertices[face.z];
        TriangleInModel { p1, p2, p3 }
    }).collect();

    let mut model = Model {
        triangles,
        translate: Vec3::new(0.0, 0.0, 0.0),
        rotate: Mat3::from_rotation_x(0.0),
        scale: 100.0
    };

    let mut image_buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut coverage_buffer: Vec<f32> = vec![0f32; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    
    // 初始化相机
    let mut camera = Camera {
        translate: Vec3::new(0.0, -10.0, -30.0),
        rotate: Mat3::from_rotation_x(0.0),
        fov: 60.0f32.to_radians(),
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
        transform: Mat4::ZERO
    };

    camera.transform = camera.perspective_transform() * camera.view_transform();

    let mut rotation = 2.0f32;

    window.set_target_fps(60);

    let mut count = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start_time = std::time::Instant::now();
        image_buffer.fill(0);
        coverage_buffer.fill(0.0);

        rotation += 0.1;  // 减慢旋转速度
        model.rotate = Mat3::from_rotation_x(std::f32::consts::PI) * Mat3::from_rotation_y(rotation);

        // 更新旋转角度

        // 渲染所有三角形
        for triangle in model.transform_triangles_iter() {
            draw_a_triangle_in_model(&triangle, &camera, &mut coverage_buffer, WIDTH, HEIGHT);
        }

        // draw_a_triangle(TriangleInScreen { p1: Vector2::new(0.0, 0.0), p2: Vector2::new(WIDTH as f32 / 2.0, HEIGHT as f32), p3: Vector2::new(WIDTH as f32, 0.0) }, &mut coverage_buffer);
        // let size = 200.0;
        // draw_a_triangle(TriangleInScreen { p1: Vector3::new(0.0, 0.0, 1.0), p2: Vector3::new(0.0, size, 1.0), p3: Vector3::new(size, 0.0, 1.0) }, &mut coverage_buffer);

        // for (i, coverage) in coverage_buffer.iter().enumerate() {
        //     // image_buffer[i] = u32::from_be_bytes([(255f32 * coverage.min(1.0)) as u8, 0, 0, 255]);
        //     image_buffer[i] = u32::from_be_bytes([0, 255, (255f32 * coverage.min(1.0)) as u8, 255]);
        // }

        let end_time = std::time::Instant::now();
        println!("Frame {} Time taken: {:?} rotation: {} triangles {}", count, end_time.duration_since(start_time), rotation, model.triangles.len());

        count += 1;

        window
            .update_with_buffer(&image_buffer, WIDTH, HEIGHT)
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });
    }
}
type Point = Vec3;

struct TriangleInModel {
    p1: Point,
    p2: Point,
    p3: Point
}

struct TriangleInScreen {
    p1: Point,
    p2: Point,
    p3: Point
}

type Transform = Mat4;

#[derive(Clone)]
struct Camera {
    translate: Vec3,
    rotate: Mat3,
    fov: f32,
    aspect: f32,
    near: f32,
    far: f32,
    transform: Transform
}

impl Camera {
    fn view_transform(&self) -> Transform {
        let translate = Mat4::from_translation(-self.translate);
        let mut rotate = Mat4::from_mat3(self.rotate.transpose());
        rotate.w_axis.w = 1.0;
        translate * rotate
    }

    fn perspective_transform(&self) -> Transform {
        let tan = (self.fov / 2.0).tan();
        let a11 = 1.0 / (self.aspect * tan);
        let a22 = 1.0 / tan;
        let a33 = (self.far + self.near) / (self.near - self.far);
        let a34 = (2.0 * self.near * self.far) / (self.near - self.far);

        // Column-major order
        Mat4::from_cols_array_2d(&[
            [a11, 0.0, 0.0, 0.0],
            [0.0, a22, 0.0, 0.0],
            [0.0, 0.0, a33, -1.0],
            [0.0, 0.0, a34, 0.0]
        ])
    }

    fn transform(&self) -> &Transform {
        &self.transform
        // self.view_transform()
        // self.perspective_transform()
        // self.perspective_transform() * self.view_transform()
    }
}

struct Rect {
    origin: Vec2,
    width: f32,
    height: f32
}

impl TriangleInScreen {
    fn bounding_box(&self) -> Rect {
        let x = self.p1.x.min(self.p2.x).min(self.p3.x);
        let y = self.p1.y.min(self.p2.y).min(self.p3.y);

        let max_x = self.p1.x.max(self.p2.x).max(self.p3.x);
        let max_y = self.p1.y.max(self.p2.y).max(self.p3.y);

        let width = max_x - x;
        let height = max_y - y;
        Rect { origin: Vec2::new(x, y), width, height }
    }

    fn test_p(&self, x: f32, y: f32) -> bool {
        let edge1 = Vec2::new(self.p2.x - self.p1.x, self.p2.y - self.p1.y);
        let edge2 = Vec2::new(self.p3.x - self.p2.x, self.p3.y - self.p2.y);
        let edge3 = Vec2::new(self.p1.x - self.p3.x, self.p1.y - self.p3.y);

        let test_edge1 = Vec2::new(x - self.p1.x, y - self.p1.y);
        let test_edge2 = Vec2::new(x - self.p2.x, y - self.p2.y);
        let test_edge3 = Vec2::new(x - self.p3.x, y - self.p3.y);

        let cross1 = edge1.perp_dot(test_edge1);
        let cross2 = edge2.perp_dot(test_edge2);
        let cross3 = edge3.perp_dot(test_edge3);
        
        if cross1 <= 0.0 && cross2 <= 0.0 && cross3 <= 0.0 {
            return true;
        }

        if cross1 >= 0.0 && cross2 >= 0.0 && cross3 >= 0.0 {
            return true;
        }

        return false;
    }

    fn coverage(&self, point: Vec2) -> f32 {
        let sample_point = Vec2::new(point.x + 0.5, point.y + 0.5);
        return self.test_p(sample_point.x, sample_point.y) as u8 as f32;
        // 使用标准的MSAA 4x采样模式
        // 采样点均匀分布在像素区域内，避免边缘重叠
        // let sample_point1 = Vec2::new(point.x + 0.375, point.y + 0.125);
        // let sample_point2 = Vec2::new(point.x + 0.875, point.y + 0.375);
        // let sample_point3 = Vec2::new(point.x + 0.125, point.y + 0.625);
        // let sample_point4 = Vec2::new(point.x + 0.625, point.y + 0.875);

        // let sample_point1_in = self.test_p(sample_point1.x, sample_point1.y);
        // let sample_point2_in = self.test_p(sample_point2.x, sample_point2.y);
        // let sample_point3_in = self.test_p(sample_point3.x, sample_point3.y);
        // let sample_point4_in = self.test_p(sample_point4.x, sample_point4.y);

        // let rate = (sample_point1_in as u8 as f32 + sample_point2_in as u8 as f32 + sample_point3_in as u8 as f32 + sample_point4_in as u8 as f32) / 4.0;
        // rate
    }
}

fn draw_a_triangle_in_model(triangle: &TriangleInModel, camera: &Camera, image: &mut Vec<f32>, width: usize, height: usize) {
    let transform = camera.transform();
    let p1_4d = Vec4::new(triangle.p1.x, triangle.p1.y, triangle.p1.z, 1.0);
    let p2_4d = Vec4::new(triangle.p2.x, triangle.p2.y, triangle.p2.z, 1.0);
    let p3_4d = Vec4::new(triangle.p3.x, triangle.p3.y, triangle.p3.z, 1.0);
    
    let p1_transformed = transform * p1_4d;
    let p2_transformed = transform * p2_4d;
    let p3_transformed = transform * p3_4d;

    let p1 = Vec3::new(p1_transformed.x / p1_transformed.w, p1_transformed.y / p1_transformed.w, p1_transformed.z / p1_transformed.w);
    let p2 = Vec3::new(p2_transformed.x / p2_transformed.w, p2_transformed.y / p2_transformed.w, p2_transformed.z / p2_transformed.w);
    let p3 = Vec3::new(p3_transformed.x / p3_transformed.w, p3_transformed.y / p3_transformed.w, p3_transformed.z / p3_transformed.w);

    let p1_screen = Vec3::new((p1.x + 1.0) * 0.5 * width as f32, (1.0 - p1.y) * 0.5 * height as f32, p1.z);
    let p2_screen = Vec3::new((p2.x + 1.0) * 0.5 * width as f32, (1.0 - p2.y) * 0.5 * height as f32, p2.z);
    let p3_screen = Vec3::new((p3.x + 1.0) * 0.5 * width as f32, (1.0 - p3.y) * 0.5 * height as f32, p3.z);

    std::hint::black_box(p1_screen);
    std::hint::black_box(p2_screen);
    std::hint::black_box(p3_screen);
    // draw_a_triangle(TriangleInScreen { p1: p1_screen, p2: p2_screen, p3: p3_screen }, image, width, height);
}

fn draw_a_triangle(triangle: TriangleInScreen, image: &mut Vec<f32>, width: usize, height: usize) {
    let mut top_point = triangle.p1;
    let mut middle_point = triangle.p2;
    let mut bottom_point = triangle.p3;

    if top_point.y > middle_point.y {
        swap(&mut top_point, &mut middle_point);
    }
    if top_point.y > bottom_point.y {
        swap(&mut top_point, &mut bottom_point);
    }
    if middle_point.y > bottom_point.y {
        swap(&mut middle_point, &mut bottom_point);
    }

    let top_y = min(max(top_point.y.floor() as i32, 0), height as i32 - 1);
    let middle_y = min(max(middle_point.y.floor() as i32, 0), height as i32 - 1);
    let bottom_y = min(max(bottom_point.y.floor() as i32, 0), height as i32 - 1);

    let slope1 = (top_point.x - middle_point.x) / (top_point.y - middle_point.y);
    let slope2 = (top_point.x - bottom_point.x) / (top_point.y - bottom_point.y);
    let slope3 = (middle_point.x - bottom_point.x) / (middle_point.y - bottom_point.y);

    if top_y != middle_y {
        for y in top_y ..= middle_y {
            let mut x1: f32 = top_point.x + slope1 * (y as f32 - top_point.y);
            let mut x2: f32 = top_point.x + slope2 * (y as f32 - top_point.y);

            if x1 > x2 {
                swap(&mut x1, &mut x2);
            }

            let lower_bounds = x1.ceil() as usize;
            let upper_bounds = x2.floor() as usize;

            for x in lower_bounds ..= upper_bounds {
                let xy = x as usize + y as usize * width;
                image[xy] += 1.0;
            }
        }
    }

    if middle_y != bottom_y {
        for y in middle_y ..= bottom_y {
            let mut x1: f32 = middle_point.x + slope3 * (y as f32 - middle_point.y);
            let mut x2: f32 = bottom_point.x + slope2 * (y as f32 - bottom_point.y);

            if x1 > x2 {
                swap(&mut x1, &mut x2);
            }

            let lower_bounds = x1.ceil() as usize;
            let upper_bounds = x2.floor() as usize;

            for x in lower_bounds ..= upper_bounds {
                let xy = x as usize + y as usize * width;
                image[xy] += 1.0;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::mem::swap;

    use glam::{Vec2, Vec3};

    #[test]
    fn test_coverage() {
        let triangle = super::TriangleInScreen { p1: Vec3::new(0.0, 0.0, 1.0), p2: Vec3::new(0.0, 2.0, 1.0), p3: Vec3::new(2.0, 0.0, 1.0) };

        let point = Vec2::new(0.0, 0.0);
        let coverage = triangle.coverage(point);
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn test_sort() {
        let mut v_1 = 3;
        let mut v_2 = 1;
        let mut v_3 = 2;

        if v_1 > v_2 {
            swap(&mut v_1, &mut v_2);
        }
        if v_1 > v_3 {
            swap(&mut v_1, &mut v_3);
        }
        if v_2 > v_3 {
            swap(&mut v_2, &mut v_3);
        }

        assert_eq!(v_1, 1);
        assert_eq!(v_2, 2);
        assert_eq!(v_3, 3);
    }

    #[test]
    fn test_draw_a_triangle() {
        let mut image = vec![0.0; 2 * 2];
        let triangle = super::TriangleInScreen { p1: Vec3::new(0.0, 0.0, 1.0), p2: Vec3::new(0.0, 1.0, 1.0), p3: Vec3::new(1.0, 0.0, 1.0) };
        super::draw_a_triangle(triangle, &mut image, 2, 2);
        assert_ne!(image[0], 0.0);
        assert_ne!(image[1], 0.0);
        assert_ne!(image[2], 0.0);
        assert_eq!(image[3], 0.0);
    }

    #[test]
    fn test_draw_a_triangle_2() {
        let mut image = vec![0.0; 2 * 2];
        let triangle = super::TriangleInScreen { p1: Vec3::new(1.0, 0.0, 1.0), p2: Vec3::new(0.0, 1.0, 1.0), p3: Vec3::new(1.5, 1.5, 1.0) };
        super::draw_a_triangle(triangle, &mut image, 2, 2);
        assert_ne!(image[0], 0.0);
        assert_ne!(image[1], 0.0);
        assert_ne!(image[2], 0.0);
        assert_ne!(image[3], 0.0);
    }

}
