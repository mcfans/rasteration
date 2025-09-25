use std::cmp::min;

use cgmath::{Matrix, Matrix3, Matrix4, Rad, Vector2, Vector3, Vector4, InnerSpace};
use minifb::{Key, Window, WindowOptions};

struct Model {
    triangles: Vec<TriangleInModel>,
    translate: Vector3<f32>,
    rotate: Matrix3<f32>,
    scale: f32
}

impl Model {
    fn transform(&self) -> Matrix4<f32> {
        let translate = Matrix4::from_translation(self.translate);
        let rotate = Matrix4::from(self.rotate);
        let scale = Matrix4::from_nonuniform_scale(self.scale, self.scale, self.scale);
        translate * rotate * scale
    }

    fn transform_triangle(&self, triangle: &TriangleInModel) -> TriangleInModel {
        let transform = self.transform();
        let p1 = transform * Vector4::new(triangle.p1.x, triangle.p1.y, triangle.p1.z, 1.0);
        let p2 = transform * Vector4::new(triangle.p2.x, triangle.p2.y, triangle.p2.z, 1.0);
        let p3 = transform * Vector4::new(triangle.p3.x, triangle.p3.y, triangle.p3.z, 1.0);

        let p1 = Vector3::new(p1.x / p1.w, p1.y / p1.w, p1.z / p1.w);
        let p2 = Vector3::new(p2.x / p2.w, p2.y / p2.w, p2.z / p2.w);
        let p3 = Vector3::new(p3.x / p3.w, p3.y / p3.w, p3.z / p3.w);

        TriangleInModel { p1, p2, p3 }
    }

    fn transform_triangles_iter(&self) -> impl Iterator<Item = TriangleInModel> + '_ {
        self.triangles.iter().map(move |triangle| self.transform_triangle(triangle))
    }
}

fn parser_file(file_path: &str) -> (Vec<Vector3<f32>>, Vec<Vector3<usize>>) {
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
            vertices.push(Vector3::new(x.unwrap(), y.unwrap(), z.unwrap()));
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
            triangles.push(Vector3::new(x.unwrap(), y.unwrap(), z.unwrap()));
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
    let (vertices, faces) = parser_file("/Users/yangxuesi/Downloads/bunny/reconstruction/bun_zipper.ply");

    let triangles: Vec<TriangleInModel> = faces.iter().map(|face| {
        let p1 = vertices[face.x];
        let p2 = vertices[face.y];
        let p3 = vertices[face.z];
        TriangleInModel { p1, p2, p3 }
    }).collect();

    let mut model = Model {
        triangles,
        translate: Vector3::new(0.0, 0.0, 0.0),
        rotate: Matrix3::from_angle_x(Rad(0.0)),
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
    let camera = Camera {
        translate: Vector3::new(0.0, 0.0, -50.0),
        rotate: Matrix3::from_angle_x(Rad(0.0)),
        fov: 60.0f32.to_radians(),
        aspect: 1.0,
        near: 0.1,
        far: 100.0
    };

    let mut rotation = 0.0f32;

    window.set_target_fps(144);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        image_buffer.fill(0);
        coverage_buffer.fill(0.0);

        // 尝试不同的旋转方式
        // 方式1: 只绕Y轴旋转（推荐）
        model.rotate = Matrix3::from_angle_y(Rad(rotation));
        
        // rotation += 0.001;  // 减慢旋转速度
        // 上下反转：绕X轴旋转180度
        model.rotate = Matrix3::from_angle_x(Rad(std::f32::consts::PI)) * model.rotate;

        // 更新旋转角度

        // 渲染所有三角形
        for triangle in model.transform_triangles_iter() {
            draw_a_triangle_in_model(&triangle, &camera, &mut coverage_buffer);
        }

        // draw_a_triangle(TriangleInScreen { p1: Vector2::new(0.0, 0.0), p2: Vector2::new(WIDTH as f32 / 2.0, HEIGHT as f32), p3: Vector2::new(WIDTH as f32, 0.0) }, &mut coverage_buffer);
        // let size = 200.0;
        // draw_a_triangle(TriangleInScreen { p1: Vector3::new(0.0, 0.0, 1.0), p2: Vector3::new(0.0, size, 1.0), p3: Vector3::new(size, 0.0, 1.0) }, &mut coverage_buffer);

        for (i, coverage) in coverage_buffer.iter().enumerate() {
            // image_buffer[i] = u32::from_be_bytes([(255f32 * coverage.min(1.0)) as u8, 0, 0, 255]);
            image_buffer[i] = u32::from_be_bytes([0, 255, (255f32 * coverage.min(1.0)) as u8, 255]);
        }

        window
            .update_with_buffer(&image_buffer, WIDTH, HEIGHT)
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });
    }
}
type Point = Vector3<f32>;

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

type Transform = Matrix4<f32>;

#[derive(Clone)]
struct Camera {
    translate: Vector3<f32>,
    rotate: Matrix3<f32>,
    fov: f32,
    aspect: f32,
    near: f32,
    far: f32
}

impl Camera {
    fn view_transform(&self) -> Transform {
        let translate = Matrix4::from_translation(-self.translate);
        let mut matrix = Matrix4::from(self.rotate.transpose());
        matrix.w.w = 1.0;
        translate * matrix
    }

    fn perspective_transform(&self) -> Transform {
        let tan = (self.fov / 2.0).tan();
        let a11 = 1.0 / (self.aspect * tan);
        let a22 = 1.0 / tan;
        let a33 = (self.far + self.near) / (self.near - self.far);
        let a34 = (2.0 * self.near * self.far) / (self.near - self.far);

        // Column-major order
        Matrix4::new(
            a11, 0.0, 0.0, 0.0,
            0.0, a22, 0.0, 0.0,
            0.0, 0.0, a33, -1.0,
            0.0, 0.0, a34, 0.0
        )
    }

    fn transform(&self) -> Transform {
        // self.view_transform()
        // self.perspective_transform()
        self.perspective_transform() * self.view_transform()
    }
}

struct Rect {
    origin: Vector2<f32>,
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
        Rect { origin: Vector2 { x, y }, width, height }
    }

    fn test_p(&self, x: f32, y: f32) -> bool {
        let edge1 = Vector2 { x: self.p2.x - self.p1.x, y: self.p2.y - self.p1.y };
        let edge2 = Vector2 { x: self.p3.x - self.p2.x, y: self.p3.y - self.p2.y };
        let edge3 = Vector2 { x: self.p1.x - self.p3.x, y: self.p1.y - self.p3.y };

        let test_edge1 = Vector2 { x: x - self.p1.x, y: y - self.p1.y };
        let test_edge2 = Vector2 { x: x - self.p2.x, y: y - self.p2.y };
        let test_edge3 = Vector2 { x: x - self.p3.x, y: y - self.p3.y };

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

    fn coverage(&self, point: Vector2<f32>) -> f32 {
        // 使用标准的MSAA 4x采样模式
        // 采样点均匀分布在像素区域内，避免边缘重叠
        let sample_point1 = Vector2 { x: point.x as f32 + 0.375, y: point.y as f32 + 0.125 };
        let sample_point2 = Vector2 { x: point.x as f32 + 0.875, y: point.y as f32 + 0.375 };
        let sample_point3 = Vector2 { x: point.x as f32 + 0.125, y: point.y as f32 + 0.625 };
        let sample_point4 = Vector2 { x: point.x as f32 + 0.625, y: point.y as f32 + 0.875 };

        let sample_point1_in = self.test_p(sample_point1.x, sample_point1.y);
        let sample_point2_in = self.test_p(sample_point2.x, sample_point2.y);
        let sample_point3_in = self.test_p(sample_point3.x, sample_point3.y);
        let sample_point4_in = self.test_p(sample_point4.x, sample_point4.y);

        let rate = (sample_point1_in as u8 as f32 + sample_point2_in as u8 as f32 + sample_point3_in as u8 as f32 + sample_point4_in as u8 as f32) / 4.0;
        rate
    }
}

fn draw_a_triangle_in_model(triangle: &TriangleInModel, camera: &Camera, image: &mut Vec<f32>) {
    // 背部剔除：计算三角形法向量和视线向量
    let edge1 = triangle.p2 - triangle.p1;
    let edge2 = triangle.p3 - triangle.p1;
    let normal = edge1.cross(edge2).normalize();
    
    // 计算三角形中心点
    let center = (triangle.p1 + triangle.p2 + triangle.p3) / 3.0;
    
    // 计算从相机到三角形中心的视线向量
    let view_direction = (center - camera.translate).normalize();
    
    // 如果法向量和视线向量的点积为负，说明三角形背对相机，应该被剔除
    if normal.dot(view_direction) < 0.0 {
        return;
    }

    let transform = camera.transform();
    let p1 = transform * Vector4::new(triangle.p1.x, triangle.p1.y, triangle.p1.z, 1.0);
    let p2 = transform * Vector4::new(triangle.p2.x, triangle.p2.y, triangle.p2.z, 1.0);
    let p3 = transform * Vector4::new(triangle.p3.x, triangle.p3.y, triangle.p3.z, 1.0);

    let p1 = Vector3::new(p1.x / p1.w, p1.y / p1.w, p1.z / p1.w);
    let p2 = Vector3::new(p2.x / p2.w, p2.y / p2.w, p2.z / p2.w);
    let p3 = Vector3::new(p3.x / p3.w, p3.y / p3.w, p3.z / p3.w);

    let p1 = Vector3 { x: (p1.x + 1.0) * 0.5 * WIDTH as f32, y: (1.0 - p1.y) * 0.5 * HEIGHT as f32, z: p1.z };
    let p2 = Vector3 { x: (p2.x + 1.0) * 0.5 * WIDTH as f32, y: (1.0 - p2.y) * 0.5 * HEIGHT as f32, z: p2.z };
    let p3 = Vector3 { x: (p3.x + 1.0) * 0.5 * WIDTH as f32, y: (1.0 - p3.y) * 0.5 * HEIGHT as f32, z: p3.z };

    draw_a_triangle(TriangleInScreen { p1, p2, p3 }, image);
}

fn draw_a_triangle(triangle: TriangleInScreen, image: &mut Vec<f32>) {
    let bounding_box = triangle.bounding_box();
    let start_x = bounding_box.origin.x.round() as i32;
    let start_y = bounding_box.origin.y.round() as i32;
    let end_x = min((bounding_box.origin.x + bounding_box.width).round() as i32, WIDTH as i32);
    let end_y = min((bounding_box.origin.y + bounding_box.height).round() as i32, HEIGHT as i32);

    for x in start_x ..= end_x {
        for y in start_y ..= end_y {
            let point = Vector2 { x: x as f32, y: y as f32 };
            let coverage = triangle.coverage(point);
            let xy = x as usize + y as usize * WIDTH;
            // if coverage >= 0.0 {
            //     image[xy] = 1.0;
            // } else {
            //     image[xy] = 0.0;
            // }
            image[xy] += coverage;
            if image[xy] >= 0.75 {
                image[xy] = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use cgmath::{Vector2, Vector3};

    #[test]
    fn test_coverage() {
        let triangle = super::TriangleInScreen { p1: Vector3::new(0.0, 0.0, 1.0), p2: Vector3::new(0.0, 2.0, 1.0), p3: Vector3::new(2.0, 0.0, 1.0) };

        let point = Vector2::new(0.0, 0.0);
        let coverage = triangle.coverage(point);
        assert_eq!(coverage, 1.0);
    }
}
