#version 330

in vec2 uv0;
uniform sampler2D texture0;
out vec4 fragColor;

uniform float c_du;
uniform float c_dv;
uniform float c_f;
uniform float c_k;

uniform float dx;
uniform float dy;
uniform float hx;
uniform float hy;
uniform float dt;

void main() {
    vec2 center = texture(texture0, uv0).xy;
    vec2 left = texture(texture0, uv0 + vec2(-dx, 0)).xy;
    vec2 right = texture(texture0, uv0 + vec2(dx, 0)).xy;
    vec2 bottom = texture(texture0, uv0 + vec2(0, -dy)).xy;
    vec2 top = texture(texture0, uv0 + vec2(0, dy)).xy;

    float lu = (left.x + right.x - 2.0 * center.x) / (hx * hx) + (top.x + bottom.x - 2.0 * center.x) / (hy * hy);
    float lv = (left.y + right.y - 2.0 * center.y) / (hx * hx) + (top.y + bottom.y - 2.0 * center.y) / (hy * hy);
    float du = c_du * lu - center.x * center.y * center.y + c_f * (1.0 - center.x);
    float dv = c_dv * lv + center.x * center.y * center.y - (c_f + c_k) * center.y;
    vec2 result = center + vec2(du, dv) * dt;

    fragColor = vec4(result, 0.0, 1.0);
}