#version 330 core
out vec4 FragColor;
out vec4 BrightColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord) * 2.0;
    float intensity = 1.0 - dist * dist;
    if (dist > 1.0) discard;

    // 更新颜色以获得更好的金色效果
    vec3 color = vec3(0.8667, 0.7, 0.2);  // 增加绿色和减少蓝色成分

    // 增强亮度和调整alpha值
    float alpha = smoothstep(0.8, 0.1, dist);
    FragColor = vec4(color * intensity, alpha);


    // 确定该片段是否应该增加到亮度纹理中
    float brightness = intensity * alpha;
    if (brightness > 0.3) {  // 降低亮度阈值以增加更多片段
        BrightColor = vec4(color * brightness, 1.0);
    } else {
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}