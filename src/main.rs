use std::cmp::min;

use cgmath::{Matrix, Matrix3, Matrix4, Rad, Vector2, Vector3, Vector4};
use pixel_canvas::{input::MouseState, Canvas, Color, Image, XY};

fn main() {
    let canvas = Canvas::new(512, 512)
        .title("3D Cube")
        .state(MouseState::new())
        .input(MouseState::handle_input);
    
    // 初始化相机
    let camera = Camera {
        translate: Vector3::new(0.0, 0.0, -50.0),
        rotate: Matrix3::from_angle_x(Rad(0.0)),
        fov: 60.0f32.to_radians(),
        aspect: 1.0,
        near: 0.1,
        far: 100.0
    };

    // 定义立方体的顶点
    let cube_vertices = [
        // 前面
        Vector3::new(-10.0, -10.0, 10.0),
        Vector3::new( 10.0, -10.0,  10.0),
        Vector3::new( 10.0,  10.0,  10.0),
        Vector3::new(-10.0,  10.0,  10.0),
        // 后面
        Vector3::new(-10.0, -10.0, -10.0),
        Vector3::new( 10.0, -10.0, -10.0),
        Vector3::new( 10.0,  10.0, -10.0),
        Vector3::new(-10.0,  10.0, -10.0),
    ];

    // 定义立方体的三角形面
    let cube_triangles = [
        // 前面
        TriangleInModel { p1: cube_vertices[0], p2: cube_vertices[1], p3: cube_vertices[2] },
        TriangleInModel { p1: cube_vertices[0], p2: cube_vertices[2], p3: cube_vertices[3] },
        // 右面
        TriangleInModel { p1: cube_vertices[1], p2: cube_vertices[5], p3: cube_vertices[6] },
        TriangleInModel { p1: cube_vertices[1], p2: cube_vertices[6], p3: cube_vertices[2] },
        // 后面
        TriangleInModel { p1: cube_vertices[5], p2: cube_vertices[4], p3: cube_vertices[7] },
        TriangleInModel { p1: cube_vertices[5], p2: cube_vertices[7], p3: cube_vertices[6] },
        // 左面
        TriangleInModel { p1: cube_vertices[4], p2: cube_vertices[0], p3: cube_vertices[3] },
        TriangleInModel { p1: cube_vertices[4], p2: cube_vertices[3], p3: cube_vertices[7] },
        // 上面
        TriangleInModel { p1: cube_vertices[3], p2: cube_vertices[2], p3: cube_vertices[6] },
        TriangleInModel { p1: cube_vertices[3], p2: cube_vertices[6], p3: cube_vertices[7] },
        // 下面
        TriangleInModel { p1: cube_vertices[4], p2: cube_vertices[5], p3: cube_vertices[1] },
        TriangleInModel { p1: cube_vertices[4], p2: cube_vertices[1], p3: cube_vertices[0] },
    ];

    let mut rotation = 0.0f32;
    
    canvas.render(move |mouse, image| {
        // 清空画布
        for pixel in image.iter_mut() {
            *pixel = Color::rgb(0, 0, 0);
        }

        // 更新旋转角度
        rotation += 0.02;
        
        // 更新相机旋转
        let mut camera = camera.clone();
        camera.rotate = Matrix3::from_angle_y(Rad(rotation)) * Matrix3::from_angle_x(Rad(rotation * 0.5));
        camera.aspect = image.width() as f32 / image.height() as f32;

        // 渲染所有三角形
        // let mut count = 0;
        for triangle in cube_triangles.iter() {
            draw_a_triangle_in_model(triangle, &camera, image);
            // count += 1;
            // if count == 1 {
            //     break;
            // }
        }
    });
}
type Point = Vector3<f32>;

struct TriangleInModel {
    p1: Point,
    p2: Point,
    p3: Point
}

struct TriangleInScreen {
    p1: Vector2<f32>,
    p2: Vector2<f32>,
    p3: Vector2<f32>
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

// #[derive(Debug, Clone, Copy)]
// struct Point {
//     x: usize,
//     y: usize,
// }

// impl Point {
//     fn new(x: usize, y: usize) -> Point {
//         return Point { x, y };
//     }
// }

// struct Rect {
//     origin: Point,
//     width: usize,
//     height: usize
// }

// struct Triangle {
//     p1: Point,
//     p2: Point,
//     p3: Point
// }

// struct Vector {
//     x: f32,
//     y: f32
// }

// impl Vector {
//     fn cross(&self, other: Vector) -> f32 {
//         self.x * other.y - self.y * other.x
//     }
// }

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
        
        if cross1 < 0.0 && cross2 < 0.0 && cross3 < 0.0 {
            return true;
        }

        if cross1 > 0.0 && cross2 > 0.0 && cross3 > 0.0 {
            return true;
        }

        return false;

        // return cross1 >= 0.0 && cross2 >= 0.0 && cross3 >= 0.0;
    }

    fn coverage(&self, point: Vector2<f32>) -> f32 {
        let sample_point1 = Vector2 { x: point.x as f32 + 0.5, y: point.y as f32 };
        let sample_point2 = Vector2 { x: point.x as f32 + 0.5, y: point.y as f32 + 1.0 };
        let sample_point3 = Vector2 { x: point.x as f32, y: point.y as f32 + 0.5 };
        let sample_point4 = Vector2 { x: point.x as f32 + 1.0, y: point.y as f32 + 0.5 };

        let sample_point1_in = self.test_p(sample_point1.x, sample_point1.y);
        let sample_point2_in = self.test_p(sample_point2.x, sample_point2.y);
        let sample_point3_in = self.test_p(sample_point3.x, sample_point3.y);
        let sample_point4_in = self.test_p(sample_point4.x, sample_point4.y);

        let rate = (sample_point1_in as u8 + sample_point2_in as u8 + sample_point3_in as u8 + sample_point4_in as u8) / 4;
        rate as f32
    }
}

fn draw_a_triangle_in_model(triangle: &TriangleInModel, camera: &Camera, image: &mut Image) {
    let transform = camera.transform();
    let p1 = transform * Vector4::new(triangle.p1.x, triangle.p1.y, triangle.p1.z, 1.0);
    let p2 = transform * Vector4::new(triangle.p2.x, triangle.p2.y, triangle.p2.z, 1.0);
    let p3 = transform * Vector4::new(triangle.p3.x, triangle.p3.y, triangle.p3.z, 1.0);

    let p1 = Vector3::new(p1.x / p1.w, p1.y / p1.w, p1.z / p1.w);
    let p2 = Vector3::new(p2.x / p2.w, p2.y / p2.w, p2.z / p2.w);
    let p3 = Vector3::new(p3.x / p3.w, p3.y / p3.w, p3.z / p3.w);

    let p1 = Vector2 { x: (p1.x + 1.0) * 0.5 * image.width() as f32, y: (1.0 - p1.y) * 0.5 * image.height() as f32 };
    let p2 = Vector2 { x: (p2.x + 1.0) * 0.5 * image.width() as f32, y: (1.0 - p2.y) * 0.5 * image.height() as f32 };
    let p3 = Vector2 { x: (p3.x + 1.0) * 0.5 * image.width() as f32, y: (1.0 - p3.y) * 0.5 * image.height() as f32 };

    draw_a_triangle(TriangleInScreen { p1, p2, p3 }, image);
}

fn draw_a_triangle(triangle: TriangleInScreen, image: &mut Image) {
    let bounding_box = triangle.bounding_box();
    let start_x = bounding_box.origin.x.round() as i32;
    let start_y = bounding_box.origin.y.round() as i32;
    let end_x = min((bounding_box.origin.x + bounding_box.width).round() as i32, image.width() as i32);
    let end_y = min((bounding_box.origin.y + bounding_box.height).round() as i32, image.height() as i32);

    for x in start_x .. end_x {
        for y in start_y .. end_y {
            let point = Vector2 { x: x as f32, y: y as f32 };
            let coverage = triangle.coverage(point);
            let xy = XY(x as usize, y as usize);
            if image[xy].r == 0 {
                let xy = XY(x as usize, y as usize);
                image[xy] = Color::rgb((255f32 * coverage).round() as u8, 0, 0);
            }
            // if coverage > 0.0 {
            //     image[xy] = Color::rgb(255, 0, 0);
            // }
        }
    }
}

#[cfg(test)]
mod test {
    use cgmath::Vector2;

    #[test]
    fn test_coverage() {
        let p1 = Vector2 { x: 329.900848, y: 182.099152 };
        let p2 = Vector2 { x: 182.099152, y: 182.099152 };
        let p3 = Vector2 { x: 182.099152, y: 329.900848 };

        let triangle = super::TriangleInScreen { p1, p2, p3 };
        let point = Vector2 { x: 200.0, y: 200.0 };

        let coverage = triangle.coverage(point);
        assert_eq!(coverage, 1.0);
    }
}