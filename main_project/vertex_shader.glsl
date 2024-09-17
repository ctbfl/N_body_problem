// vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float pointMass; // 从顶点属性接受质量

uniform vec3 scaleFactors; // 缩放因子
uniform mat4 view; // 视图矩阵
uniform mat4 projection; // 投影矩阵

void main() {
    float newX = aPos.x / (scaleFactors.x+1); // 平移后缩放
    float newY = aPos.y / (scaleFactors.y+1);         // 假设Y轴的范围类似处理
    float newZ = aPos.z / (scaleFactors.z+1);    // 假设Z轴的范围类似处理
    vec4 scaledPos =  vec4(newX, newY, newZ, 1.0);

    gl_Position = projection *view * scaledPos;

    // 根据 pointMass 的值决定点的大小
    if (pointMass > 0.02) {
        gl_PointSize = 15.0; // 如果 pointMass 大于 0.02, 点的大小设置为 5
    } else {
        gl_PointSize = 10.0; // 否则, 点的大小设置为 2
    }
    // gl_PointSize = max(1.0f,pointMass*500.0f);
}