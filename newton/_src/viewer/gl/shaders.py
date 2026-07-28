# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes

import numpy as np

shadow_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

uniform mat4 light_space_matrix;

void main()
{
    mat4 transform = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    gl_Position = light_space_matrix * transform * vec4(aPos, 1.0);
}
"""

shadow_fragment_shader = """
#version 330 core

void main() { }
"""


shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

// colors to use for the checker_enable pattern
layout (location = 7) in vec3 aObjectColor;

// material properties
layout (location = 8) in vec4 aMaterial;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 light_space_matrix;

out vec3 Normal;
out vec3 FragPos;
out vec3 LocalPos;
out vec2 TexCoord;
out vec3 ObjectColor;
out vec4 FragPosLightSpace;
out vec4 Material;

void main()
{
    mat4 transform = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);

    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    LocalPos = aPos;

    mat3 rotation = mat3(transform);
    // transpose(inverse(...)) handles non-uniform scale. The extra sign flip for
    // det < 0 keeps shading normals outward when the viewer caches a winding-
    // flipped variant of the source mesh for mirrored instances: the winding
    // swap exposes the originally-back side of the mesh as front-facing, and
    // negating here restores the outward-pointing normal in world space.
    mat3 normalMatrix = transpose(inverse(rotation));
    if (determinant(rotation) < 0.0) normalMatrix = -normalMatrix;
    Normal = normalMatrix * aNormal;
    TexCoord = aTexCoord;
    ObjectColor = aObjectColor;
    FragPosLightSpace = light_space_matrix * worldPos;
    Material = aMaterial;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec3 LocalPos;
in vec2 TexCoord;
in vec3 ObjectColor; // used as albedo
in vec4 FragPosLightSpace;
in vec4 Material;

uniform vec3 view_pos;
#define LIGHT_COUNT 3
uniform vec3 light_dirs[LIGHT_COUNT];
uniform vec3 light_colors[LIGHT_COUNT];
uniform vec3 sky_color;
uniform vec3 ground_color;
uniform vec3 sun_direction;
uniform sampler2D shadow_map;
uniform sampler2D env_map;
uniform float env_intensity;
uniform sampler2D albedo_map;

uniform vec3 fogColor;
uniform int up_axis;

uniform mat4 light_space_matrix;

uniform float shadow_radius;
uniform float diffuse_scale;
uniform float specular_scale;
uniform bool spotlight_enabled;
uniform float shadow_extents;
uniform float shadow_resolution;
uniform float exposure;

const float PI = 3.14159265359;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Analytic filtering helpers for smooth checker_enable pattern
float filterwidth(vec2 v)
{
    vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
    return max(fw.x, fw.y);
}

vec2 bump(vec2 x)
{
    return (floor(x / 2.0) + 2.0 * max(x / 2.0 - floor(x / 2.0) - 0.5, 0.0));
}

float checker(vec2 uv)
{
    float width = filterwidth(uv);
    vec2 p0 = uv - 0.5 * width;
    vec2 p1 = uv + 0.5 * width;

    vec2 i = (bump(p1) - bump(p0)) / width;
    return i.x * i.y + (1.0 - i.x) * (1.0 - i.y);
}

vec2 poissonDisk[16] = vec2[](
   vec2( -0.94201624, -0.39906216 ),
   vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),
   vec2( 0.34495938, 0.29387760 ),
   vec2( -0.91588581, 0.45771432 ),
   vec2( -0.81544232, -0.87912464 ),
   vec2( -0.38277543, 0.27676845 ),
   vec2( 0.97484398, 0.75648379 ),
   vec2( 0.44323325, -0.97511554 ),
   vec2( 0.53742981, -0.47373420 ),
   vec2( -0.26496911, -0.41893023 ),
   vec2( 0.79197514, 0.19090188 ),
   vec2( -0.24188840, 0.99706507 ),
   vec2( -0.81409955, 0.91437590 ),
   vec2( 0.19984126, 0.78641367 ),
   vec2( 0.14383161, -0.14100790 )
);

float ShadowCalculation()
{
    vec3 normal = normalize(Normal);

    if (!gl_FrontFacing)
        normal = -normal;

    vec3 lightDir = normalize(sun_direction);

    // bias in normal dir - adjust for backfacing triangles
    float worldTexel = (shadow_extents * 2.0) / shadow_resolution; // world extent / shadow map resolution
    float normalBias = 2.0 * worldTexel;   // tune ~1-3

    // For backfacing triangles, we might need different bias handling
    vec4 light_space_pos;
    light_space_pos = light_space_matrix * vec4(FragPos + normal * normalBias, 1.0);
    vec3 projCoords = light_space_pos.xyz/light_space_pos.w;

    // map to [0,1]
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0)
        return 0.0;
    float frag_depth = projCoords.z;

    // Fade shadow to zero near edges of the shadow map to avoid hard rectangle
    float fade = 1.0;
    float margin = 0.15;
    fade *= smoothstep(0.0, margin, projCoords.x);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.x);
    fade *= smoothstep(0.0, margin, projCoords.y);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.y);

    // Slope-scaled depth bias: more bias when surface is nearly parallel to light
    // (where self-shadowing from float precision is worst), minimal when facing light.
    float NdotL_bias = max(dot(normal, lightDir), 0.0);
    float depthBias = mix(0.0003, 0.00002, NdotL_bias);
    float biased_depth = frag_depth - depthBias;

    float shadow = 0.0;
    float radius = shadow_radius;
    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    float angle = rand(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rotationMatrix = mat2(c, -s, s, c);
    for(int i = 0; i < 16; i++)
    {
        vec2 offset = rotationMatrix * poissonDisk[i];
        float pcf_depth = texture(shadow_map, projCoords.xy + offset * radius * texelSize).r;
        if(pcf_depth < biased_depth)
            shadow += 1.0;
    }
    shadow /= 16.0;
    return shadow * fade;
}

float SpotlightAttenuation()
{
    if (!spotlight_enabled)
        return 1.0;

    // Calculate spotlight position as 20 units from the camera in sun direction
    vec3 spotlight_pos = view_pos + sun_direction * 20.0;

    // Vector from fragment to spotlight
    vec3 fragToLight = normalize(spotlight_pos - FragPos);

    // Angle between spotlight direction (towards origin) and vector from light to fragment
    float cosAngle = dot(normalize(sun_direction), fragToLight);

    // Fixed cone angles (inner: 30 degrees, outer: 45 degrees)
    float cosInnerAngle = cos(radians(30.0));
    float cosOuterAngle = cos(radians(45.0));

    // Smooth falloff between inner and outer cone
    float intensity = smoothstep(cosOuterAngle, cosInnerAngle, cosAngle);

    return intensity;
}

vec3 sample_env_map(vec3 dir, float lod)
{
    // dir assumed normalized
    // Convert to a Y-up reference frame before equirect sampling.
    vec3 dir_up = dir;
    if (up_axis == 0) {
        dir_up = vec3(-dir.y, dir.x, dir.z); // X-up -> Y-up
    } else if (up_axis == 2) {
        dir_up = vec3(dir.x, dir.z, -dir.y); // Z-up -> Y-up
    }
    float u = atan(dir_up.z, dir_up.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir_up.y, -1.0, 1.0)) / PI + 0.5;
    return textureLod(env_map, vec2(u, v), lod).rgb;
}

void main()
{
    // material properties from vertex shader
    float roughness = clamp(Material.x, 0.0, 1.0);
    float metallic = clamp(Material.y, 0.0, 1.0);
    float checker_enable = Material.z;
    float texture_enable = Material.w;
    float checker_scale = 1.0;

    // convert to linear space
    vec3 albedo = pow(ObjectColor, vec3(2.2));
    if (texture_enable > 0.5)
    {
        vec3 tex_color = texture(albedo_map, TexCoord).rgb;
        albedo *= pow(tex_color, vec3(2.2));
    }

    // Optional checker pattern in object-space so it follows instance transforms
    if (checker_enable > 0.0)
    {
        vec2 uv = LocalPos.xy * checker_scale;
        float cb = checker(uv);
        vec3 albedo2 = albedo*0.7;
        // pick between the two colors
        albedo = mix(albedo, albedo2, cb);
    }

    // Specular color: dielectrics ~0.04, metals use albedo.
    // Computed before desaturation so F0 reflects true material reflectance.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Metals appear paler/desaturated because their look is dominated by
    // bright specular reflections.  Without full IBL we approximate this by
    // lifting the albedo toward a brighter, less saturated version.
    float luma = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    albedo = mix(albedo, vec3(luma * 1.4), metallic * 0.45);

    // surface vectors
    vec3 N = normalize(Normal);
    vec3 V = normalize(view_pos - FragPos);
    // Flip normal for backfacing triangles
    if (!gl_FrontFacing) N = -N;
    float NdotV = max(dot(N, V), 0.001);

    // Terms shared by every light.
    float a = roughness * roughness;
    float a2 = a * a;
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    // Schlick Fresnel, dampened by roughness to reduce edge aliasing
    vec3 F_max = mix(F0, vec3(1.0), 1.0 - roughness);

    // shadows (only light 0 casts: a single shadow map represents what it sees)
    float shadow = ShadowCalculation();

    // Direct lighting, summed over the lookdev key/fill/rim lights. With a
    // single light this reduces to the pre-lookdev Cook-Torrance term.
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < LIGHT_COUNT; ++i)
    {
        vec3 L = normalize(light_dirs[i]);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;
        vec3 H = normalize(V + L);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        // GGX/Trowbridge-Reitz normal distribution
        float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = a2 / (PI * denom * denom);
        // Schlick-GGX geometry function (Smith method for both view and light)
        float G1_L = NdotL / (NdotL * (1.0 - k) + k);
        float G = G1_V * G1_L;
        vec3 F = F0 + (F_max - F0) * pow(1.0 - HdotV, 5.0);
        // Cook-Torrance specular BRDF
        vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        // Diffuse uses remaining energy not reflected
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo / PI;

        float lit = (i == 0) ? (1.0 - shadow) : 1.0;
        Lo += (diffuse * diffuse_scale + spec * specular_scale) * light_colors[i] * NdotL * 3.0 * lit;
    }

    // Hemispherical ambient (kept subtle for depth)
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (up_axis == 0) up = vec3(1.0, 0.0, 0.0);
    if (up_axis == 2) up = vec3(0.0, 0.0, 1.0);
    float sky_fac = dot(N, up) * 0.5 + 0.5;
    vec3 ambient = mix(ground_color, sky_color, sky_fac) * albedo * 0.7;
    // Fresnel-weighted ambient specular — only significant for metals
    // (dielectrics need a prefiltered IBL for correct ambient specular)
    vec3 F_ambient = F0 + (F_max - F0) * pow(1.0 - NdotV, 5.0);
    vec3 kD_ambient = (1.0 - F_ambient) * (1.0 - metallic);
    vec3 ambient_spec = F_ambient * mix(ground_color, sky_color, sky_fac) * 0.35;
    ambient = kD_ambient * ambient + ambient_spec * metallic;

    float spotAttenuation = SpotlightAttenuation();
    vec3 color = ambient + spotAttenuation * Lo;

    // Environment / image-based lighting for metals
    vec3 R = reflect(-V, N);
    float env_lod = roughness * 8.0;
    vec3 env_color = pow(sample_env_map(R, env_lod), vec3(2.2));
    vec3 env_F = F0 + (F_max - F0) * pow(1.0 - NdotV, 5.0);
    vec3 env_spec = env_color * env_F * env_intensity;
    color += env_spec * metallic;

    // fog
    float dist = length(FragPos - view_pos);
    float fog_start = 20.0;
    float fog_end   = 200.0;
    float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    color = mix(color, pow(fogColor, vec3(2.2)), fog_factor);

    // ACES filmic tone mapping
    color = color * exposure;
    vec3 x = color;
    color = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    // gamma correction (sRGB)
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
"""


sky_vertex_shader = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 view_pos;

uniform float far_plane;

out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = vec4(aPos * far_plane + view_pos, 1.0);
    gl_Position = projection * view * worldPos;

    FragPos = vec3(worldPos);
    TexCoord = aTexCoord;
}
"""

sky_fragment_shader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 view_pos;
uniform vec3 sky_upper;
uniform vec3 sky_lower;
uniform float far_plane;
uniform float exposure;

uniform vec3 sun_direction;
uniform int up_axis;
uniform bool neutral_sky;

void main()
{
    // Vertical two-color gradient (cyclorama-style). No sun disk: the look
    // is a clean studio backdrop, not a procedural daylight sky.
    //
    // Map height onto the *full* sphere so the gradient runs continuously
    // from nadir (0.0) through horizon (0.5) to zenith (1.0). Without
    // this the lower hemisphere would be a flat ``sky_lower`` colour,
    // producing a visible step at the horizon where the analytical
    // ground meets the sky. Half-sphere mapping is what made the
    // background look "solid": much of the visible FOV at horizon-
    // grazing angles falls below the half-sphere clamp.
    vec3 dir = normalize(FragPos - view_pos);
    float h = up_axis == 0 ? dir.x : (up_axis == 1 ? dir.y : dir.z);
    float h_sphere = up_axis == 0 ? FragPos.x : (up_axis == 1 ? FragPos.y : FragPos.z);

    // Lookdev-off reproduces the pre-lookdev viewer's sky exactly: a plain
    // linear sRGB gradient from ``sky_lower`` at the horizon (h<=0) to
    // ``sky_upper`` at the zenith, with no tone mapping (matches main's
    // ShaderSky). The tinted presets fall through to the ACES path below.
    if (neutral_sky) {
        // Reproduce the pre-lookdev sky exactly: the same height ramp (measured
        // on the sky sphere, not the view ray) and the same procedural sun disk,
        // so lookdev-off keeps the viewer's original backdrop. Dithering is the
        // only addition -- it removes 8-bit banding without shifting any colour.
        float height = max(0.0, h_sphere / far_plane);
        vec3 sky = mix(sky_lower, sky_upper, height);
        float sun_diff = max(dot(sun_direction, normalize(FragPos)), 0.0);
        vec3 sun = pow(sun_diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;
        vec3 out_col = sky + sun;
        out_col += (fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))
            + fract(52.9829189 * fract(dot(gl_FragCoord.xy + 71.0, vec2(0.06711056, 0.00583715)))) - 1.0) / 255.0;
        FragColor = vec4(out_col, 1.0);
        return;
    }

    float t_norm = clamp(h * 0.5 + 0.5, 0.0, 1.0);

    // ``sky_upper`` / ``sky_lower`` are user-facing display-sRGB
    // colours. Convert to linear before mixing so the gradient is
    // computed in radiance space, then run the same exposure + ACES +
    // sRGB encode the ground shader applies. Without this, the sky
    // skips tone mapping while the ground does not, producing a hard
    // brightness/saturation seam at the horizon.
    vec3 sky_lo = pow(max(sky_lower, vec3(0.0)), vec3(2.2));
    vec3 sky_up = pow(max(sky_upper, vec3(0.0)), vec3(2.2));
    vec3 color = mix(sky_lo, sky_up, t_norm);

    color = color * exposure;
    color = (color * (2.51 * color + 0.03)) /
            (color * (2.43 * color + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);
    color = pow(color, vec3(1.0 / 2.2));
    color += (fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))
            + fract(52.9829189 * fract(dot(gl_FragCoord.xy + 71.0, vec2(0.06711056, 0.00583715)))) - 1.0) / 255.0;
    FragColor = vec4(color, 1.0);
}
"""

frame_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

frame_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture_sampler;

void main() {
    FragColor = texture(texture_sampler, TexCoord);
}
"""


def str_buffer(string: str):
    """Convert string to C-style char pointer for OpenGL."""
    return ctypes.c_char_p(string.encode("utf-8"))


def arr_pointer(arr: np.ndarray):
    """Convert numpy array to C-style float pointer for OpenGL."""
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class ShaderGL:
    """Base class for OpenGL shader wrappers."""

    def __init__(self):
        self.shader_program = None
        self._gl = None

    def _get_uniform_location(self, name: str):
        """Get uniform location for given name."""
        if self.shader_program is None:
            raise RuntimeError("Shader not initialized")
        return self._gl.glGetUniformLocation(self.shader_program.id, str_buffer(name))

    def use(self):
        """Bind this shader for use."""
        if self.shader_program is None:
            raise RuntimeError("Shader not initialized")
        self._gl.glUseProgram(self.shader_program.id)

    def __enter__(self):
        """Context manager entry - bind shader."""
        self.use()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass  # OpenGL doesn't need explicit unbinding


class ShaderShape(ShaderGL):
    """Shader for rendering 3D shapes with lighting and shadows."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(shape_fragment_shader, "fragment")
        )

        # Get all uniform locations
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_view_pos = self._get_uniform_location("view_pos")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_shadow_map = self._get_uniform_location("shadow_map")
            self.loc_albedo_map = self._get_uniform_location("albedo_map")
            self.loc_env_map = self._get_uniform_location("env_map")
            self.loc_env_intensity = self._get_uniform_location("env_intensity")
            self.loc_fog_color = self._get_uniform_location("fogColor")
            self.loc_up_axis = self._get_uniform_location("up_axis")
            self.loc_sun_direction = self._get_uniform_location("sun_direction")
            self.loc_light_dirs = self._get_uniform_location("light_dirs")
            self.loc_light_colors = self._get_uniform_location("light_colors")
            self.loc_ground_color = self._get_uniform_location("ground_color")
            self.loc_sky_color = self._get_uniform_location("sky_color")
            self.loc_shadow_radius = self._get_uniform_location("shadow_radius")
            self.loc_diffuse_scale = self._get_uniform_location("diffuse_scale")
            self.loc_specular_scale = self._get_uniform_location("specular_scale")
            self.loc_spotlight_enabled = self._get_uniform_location("spotlight_enabled")
            self.loc_shadow_extents = self._get_uniform_location("shadow_extents")
            self.loc_shadow_resolution = self._get_uniform_location("shadow_resolution")
            self.loc_exposure = self._get_uniform_location("exposure")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        view_pos: tuple[float, float, float],
        fog_color: tuple[float, float, float],
        up_axis: int,
        sun_direction: tuple[float, float, float],
        light_dirs: np.ndarray | None = None,
        light_colors: np.ndarray | None = None,
        ground_color: tuple[float, float, float] = (0.3, 0.3, 0.35),
        sky_color: tuple[float, float, float] = (0.8, 0.8, 0.85),
        enable_shadows: bool = False,
        shadow_texture: int | None = None,
        light_space_matrix: np.ndarray | None = None,
        env_texture: int | None = None,
        env_intensity: float = 1.0,
        shadow_radius: float = 3.0,
        diffuse_scale: float = 1.0,
        specular_scale: float = 1.0,
        spotlight_enabled: bool = True,
        shadow_extents: float = 10.0,
        shadow_resolution: float = 2048.0,
        exposure: float = 1.6,
    ):
        """Update all shader uniforms."""
        with self:
            # Basic matrices
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform3f(self.loc_view_pos, *view_pos)

            # Lighting
            self._gl.glUniform3f(self.loc_sun_direction, *sun_direction)
            if light_dirs is not None:
                self._gl.glUniform3fv(self.loc_light_dirs, 3, arr_pointer(light_dirs))
            if light_colors is not None:
                self._gl.glUniform3fv(self.loc_light_colors, 3, arr_pointer(light_colors))
            self._gl.glUniform3f(self.loc_ground_color, *ground_color)
            self._gl.glUniform3f(self.loc_sky_color, *sky_color)
            self._gl.glUniform1f(self.loc_shadow_radius, shadow_radius)
            self._gl.glUniform1f(self.loc_diffuse_scale, diffuse_scale)
            self._gl.glUniform1f(self.loc_specular_scale, specular_scale)
            self._gl.glUniform1i(self.loc_spotlight_enabled, int(spotlight_enabled))
            self._gl.glUniform1f(self.loc_shadow_extents, shadow_extents)
            self._gl.glUniform1f(self.loc_shadow_resolution, float(shadow_resolution))
            self._gl.glUniform1f(self.loc_exposure, exposure)

            # Fog and rendering options
            self._gl.glUniform3f(self.loc_fog_color, *fog_color)
            self._gl.glUniform1i(self.loc_up_axis, up_axis)

            # Shadows
            # if enable_shadows and shadow_texture is not None and light_space_matrix is not None:
            self._gl.glActiveTexture(self._gl.GL_TEXTURE0)
            self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, shadow_texture)
            self._gl.glUniform1i(self.loc_shadow_map, 0)
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )
            self._gl.glUniform1i(self.loc_albedo_map, 1)
            self._gl.glActiveTexture(self._gl.GL_TEXTURE2)
            if env_texture is not None:
                self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, env_texture)
            else:
                from .opengl import RendererGL  # noqa: PLC0415

                self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())
            self._gl.glUniform1i(self.loc_env_map, 2)
            self._gl.glUniform1f(self.loc_env_intensity, float(env_intensity))


class ShaderSky(ShaderGL):
    """Shader for rendering sky background."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(sky_vertex_shader, "vertex"), Shader(sky_fragment_shader, "fragment")
        )

        # Get all uniform locations
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_sky_upper = self._get_uniform_location("sky_upper")
            self.loc_sky_lower = self._get_uniform_location("sky_lower")
            self.loc_far_plane = self._get_uniform_location("far_plane")
            self.loc_view_pos = self._get_uniform_location("view_pos")
            self.loc_up_axis = self._get_uniform_location("up_axis")
            self.loc_exposure = self._get_uniform_location("exposure")
            self.loc_neutral_sky = self._get_uniform_location("neutral_sky")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_pos: tuple[float, float, float],
        camera_far: float,
        sky_upper: tuple[float, float, float],
        sky_lower: tuple[float, float, float],
        up_axis: int = 2,
        exposure: float = 1.0,
        neutral_sky: bool = False,
    ):
        """Update all shader uniforms."""
        with self:
            # Matrices and view position
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform3f(self.loc_view_pos, *camera_pos)
            self._gl.glUniform1f(self.loc_far_plane, camera_far * 0.9)  # moves sphere slightly inside far clip plane

            # Sky colors and settings
            self._gl.glUniform3f(self.loc_sky_upper, *sky_upper)
            self._gl.glUniform3f(self.loc_sky_lower, *sky_lower)
            self._gl.glUniform1i(self.loc_up_axis, up_axis)
            self._gl.glUniform1f(self.loc_exposure, float(exposure))
            self._gl.glUniform1i(self.loc_neutral_sky, 1 if neutral_sky else 0)


class ShadowShader(ShaderGL):
    """Shader for rendering shadow maps."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shadow_vertex_shader, "vertex"), Shader(shadow_fragment_shader, "fragment")
        )

        # Get uniform locations
        with self:
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")

    def update(self, light_space_matrix: np.ndarray):
        """Update light space matrix for shadow rendering."""
        with self:
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )


class FrameShader(ShaderGL):
    """Shader for rendering the final frame buffer to screen."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(frame_fragment_shader, "fragment")
        )

        # Get uniform locations
        with self:
            self.loc_texture = self._get_uniform_location("texture_sampler")

    def update(self, texture_unit: int = 0):
        """Update texture uniform."""
        with self:
            self._gl.glUniform1i(self.loc_texture, texture_unit)


wireframe_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 world;

out vec3 vertexColor;

void main()
{
    vec4 worldPos = world * vec4(aPos, 1.0);
    vertexColor = aColor;
    gl_Position = projection * view * worldPos;
}
"""

wireframe_geometry_shader = """
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;

in vec3 vertexColor[2];

out vec3 lineColor;

uniform float inv_asp_ratio;
uniform float line_width;

void main()
{
    vec4 s = gl_in[0].gl_Position;
    vec4 e = gl_in[1].gl_Position;

    if (s.w <= 0.0 || e.w <= 0.0) return;

    vec2 s_ndc = s.xy / s.w;
    vec2 e_ndc = e.xy / e.w;
    float s_depth = s.z / s.w;
    float e_depth = e.z / e.w;

    // Compute perpendicular in screen (aspect-corrected) space so line
    // width is uniform on non-square viewports.
    float safe_asp = max(inv_asp_ratio, 1e-6);
    vec2 dir_ndc = e_ndc - s_ndc;
    vec2 dir_scr = vec2(dir_ndc.x / safe_asp, dir_ndc.y);
    vec2 right_scr = normalize(vec2(dir_scr.y, -dir_scr.x));
    vec2 right = vec2(right_scr.x * safe_asp, right_scr.y);

    vec3 color = 0.5 * (vertexColor[0] + vertexColor[1]);
    vec2 xy = 0.5 * line_width * right;

    gl_Position = vec4(s_ndc - xy, s_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc + xy, e_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(s_ndc + xy, s_depth, 1); lineColor = color;
    EmitVertex();
    EndPrimitive();

    gl_Position = vec4(s_ndc - xy, s_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc - xy, e_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc + xy, e_depth, 1); lineColor = color;
    EmitVertex();
    EndPrimitive();
}
"""

wireframe_fragment_shader = """
#version 330 core
in vec3 lineColor;
out vec4 FragColor;

uniform float alpha;

void main()
{
    FragColor = vec4(lineColor, alpha);
}
"""


class ShaderLine(ShaderGL):
    """Geometry-shader-based line renderer that expands GL_LINES into screen-space quads."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(wireframe_vertex_shader, "vertex"),
            Shader(wireframe_geometry_shader, "geometry"),
            Shader(wireframe_fragment_shader, "fragment"),
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_world = self._get_uniform_location("world")
            self.loc_inv_asp_ratio = self._get_uniform_location("inv_asp_ratio")
            self.loc_line_width = self._get_uniform_location("line_width")
            self.loc_alpha = self._get_uniform_location("alpha")

    def update_frame(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_asp_ratio: float,
        line_width: float = 0.003,
        alpha: float = 0.7,
    ):
        """Set per-frame uniforms (call once before rendering all wireframe shapes)."""
        self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
        self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
        self._gl.glUniform1f(self.loc_inv_asp_ratio, float(inv_asp_ratio))
        self._gl.glUniform1f(self.loc_line_width, float(line_width))
        self._gl.glUniform1f(self.loc_alpha, float(alpha))

    def set_world(self, world: np.ndarray):
        """Set the per-shape world matrix uniform."""
        self._gl.glUniformMatrix4fv(self.loc_world, 1, self._gl.GL_FALSE, arr_pointer(world))


arrow_geometry_shader = """
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 9) out;

in vec3 vertexColor[2];
out vec3 lineColor;

uniform float inv_asp_ratio;
uniform float line_width;
uniform float arrow_size;

void main()
{
    vec4 s = gl_in[0].gl_Position;
    vec4 e = gl_in[1].gl_Position;
    if (s.w <= 0.0 || e.w <= 0.0) return;

    vec2 s_ndc = s.xy / s.w;
    vec2 e_ndc = e.xy / e.w;
    float s_depth = s.z / s.w;
    float e_depth = e.z / e.w;

    // Work in screen space (aspect-corrected) so arrows look correct on
    // non-square viewports.  screen_x = ndc_x / inv_asp_ratio.
    float safe_asp = max(inv_asp_ratio, 1e-6);
    vec2 dir_ndc = e_ndc - s_ndc;
    vec2 dir_scr = vec2(dir_ndc.x / safe_asp, dir_ndc.y);
    float len = length(dir_scr);

    vec3 color = 0.5 * (vertexColor[0] + vertexColor[1]);

    // Degenerate case: line points into/out of screen
    if (len < 1e-6) {
        float r = arrow_size * 0.4;
        vec2 up = vec2(0.0, r);
        vec2 rt = vec2(r * safe_asp, 0.0);
        gl_Position = vec4(e_ndc + up, e_depth, 1); lineColor = color; EmitVertex();
        gl_Position = vec4(e_ndc - rt, e_depth, 1); lineColor = color; EmitVertex();
        gl_Position = vec4(e_ndc + rt, e_depth, 1); lineColor = color; EmitVertex();
        EndPrimitive();
        return;
    }

    // fwd/right in screen space, then convert offsets back to NDC (scale x by safe_asp)
    vec2 fwd_scr = dir_scr / len;
    vec2 right_scr = vec2(fwd_scr.y, -fwd_scr.x);
    vec2 fwd   = vec2(fwd_scr.x * safe_asp, fwd_scr.y);
    vec2 right = vec2(right_scr.x * safe_asp, right_scr.y);

    // Shorten the line body so it ends at the arrowhead base
    vec2 xy = 0.5 * line_width * right;
    vec2 e_body = e_ndc - fwd * arrow_size;

    gl_Position = vec4(s_ndc  - xy, s_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body + xy, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(s_ndc  + xy, s_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();

    gl_Position = vec4(s_ndc  - xy, s_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body - xy, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body + xy, e_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();

    // Triangle 3: arrowhead with tip exactly at the endpoint
    vec2 tip    = e_ndc;
    vec2 base_l = e_body - right * arrow_size * 0.5;
    vec2 base_r = e_body + right * arrow_size * 0.5;

    gl_Position = vec4(tip,    e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(base_l, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(base_r, e_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();
}
"""


class ShaderArrow(ShaderGL):
    """Geometry-shader-based arrow renderer: wide line + arrowhead triangle per segment."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(wireframe_vertex_shader, "vertex"),
            Shader(arrow_geometry_shader, "geometry"),
            Shader(wireframe_fragment_shader, "fragment"),
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_world = self._get_uniform_location("world")
            self.loc_inv_asp_ratio = self._get_uniform_location("inv_asp_ratio")
            self.loc_line_width = self._get_uniform_location("line_width")
            self.loc_arrow_size = self._get_uniform_location("arrow_size")
            self.loc_alpha = self._get_uniform_location("alpha")

    def update_frame(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_asp_ratio: float,
        line_width: float = 0.003,
        arrow_size: float = 0.01,
        alpha: float = 1.0,
    ):
        """Set per-frame uniforms (call once before rendering all arrow batches)."""
        self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
        self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
        self._gl.glUniform1f(self.loc_inv_asp_ratio, float(inv_asp_ratio))
        self._gl.glUniform1f(self.loc_line_width, float(line_width))
        self._gl.glUniform1f(self.loc_arrow_size, float(arrow_size))
        self._gl.glUniform1f(self.loc_alpha, float(alpha))

    def set_world(self, world: np.ndarray):
        """Set the per-shape world matrix uniform."""
        self._gl.glUniformMatrix4fv(self.loc_world, 1, self._gl.GL_FALSE, arr_pointer(world))


edge_fragment_shader = """
#version 330 core
out vec4 FragColor;
uniform vec4 edge_color;
void main()
{
    FragColor = edge_color;
}
"""


# ---------------------------------------------------------------------------
# Analytical ground (shadow catcher) shader
# ---------------------------------------------------------------------------

ground_vertex_shader = """
#version 330 core

// Synthesises a fullscreen triangle (no VBO) and reconstructs a world-space
// view ray for each fragment via the inverse view-projection matrix. This is
// scale-invariant by construction: the ray is rebuilt from clip-space normals,
// so the floor extends to the horizon at any camera distance.

uniform mat4 inv_view_proj;
uniform vec3 view_pos;

out vec3 RayDirWorld;

void main() {
    // gl_VertexID 0..2 -> a fullscreen triangle covering [-1,1]^2 in NDC.
    vec2 ndc = vec2(
        (gl_VertexID == 1) ? 3.0 : -1.0,
        (gl_VertexID == 2) ? 3.0 : -1.0
    );
    gl_Position = vec4(ndc, 1.0, 1.0);

    vec4 worldFar = inv_view_proj * vec4(ndc, 1.0, 1.0);
    worldFar /= worldFar.w;
    RayDirWorld = worldFar.xyz - view_pos;
}
"""

ground_fragment_shader = """
#version 330 core

in vec3 RayDirWorld;
out vec4 FragColor;

uniform vec3 view_pos;
uniform mat4 view_proj;
uniform mat4 light_space_matrix;
uniform int up_axis;

uniform vec3 plane_normal;
uniform float plane_offset;

uniform vec3 ground_color;
uniform float ground_roughness;

uniform vec3 sun_direction;
uniform vec3 light_dirs[3];
uniform vec3 light_colors[3];

uniform sampler2D shadow_map;
uniform float shadow_radius;
uniform float shadow_extents;
uniform float shadow_resolution;

uniform vec3 ambient_sky_color;
uniform vec3 ambient_ground_color;

// Sky gradient endpoints (display-sRGB) — used for the Fresnel horizon
// blend that fades the floor into the sky at grazing angles. Mirrors
// the sky shader's full-sphere mapping so there is no horizon seam.
uniform vec3 sky_upper;
uniform vec3 sky_lower;

uniform float exposure;

const float PI = 3.14159265359;

vec3 to_yup(vec3 dir) {
    if (up_axis == 0) return vec3(-dir.y, dir.x, dir.z);
    if (up_axis == 2) return vec3(dir.x, dir.z, -dir.y);
    return dir;
}

float rand2(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 poissonDisk16[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

float ground_shadow(vec3 fragPos, vec3 normal) {
    vec3 lightDir = normalize(sun_direction);
    float worldTexel = (shadow_extents * 2.0) / shadow_resolution;
    float normalBias = 2.0 * worldTexel;
    vec4 lightSpace = light_space_matrix * vec4(fragPos + normal * normalBias, 1.0);
    vec3 projCoords = lightSpace.xyz / lightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0) return 0.0;

    float fade = 1.0;
    float margin = 0.15;
    fade *= smoothstep(0.0, margin, projCoords.x);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.x);
    fade *= smoothstep(0.0, margin, projCoords.y);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.y);

    float NdotL_bias = max(dot(normal, lightDir), 0.0);
    float depthBias = mix(0.0003, 0.00002, NdotL_bias);
    float biased_depth = projCoords.z - depthBias;

    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    float angle = rand2(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 offset = rot * poissonDisk16[i];
        float pcf_depth = texture(shadow_map, projCoords.xy + offset * shadow_radius * texelSize).r;
        if (pcf_depth < biased_depth) shadow += 1.0;
    }
    shadow /= 16.0;
    return shadow * fade;
}

uniform bool spotlight_enabled;
uniform bool shadow_catcher;

// Spotlight cone, matching the shape shader so the floor and the objects on it
// fall off together. Lookdev presets disable it (they use their own rig).
float SpotlightAttenuation(vec3 fragPos) {
    if (!spotlight_enabled) return 1.0;
    vec3 spotlight_pos = view_pos + sun_direction * 20.0;
    vec3 fragToLight = normalize(spotlight_pos - fragPos);
    float cosAngle = dot(normalize(sun_direction), fragToLight);
    return smoothstep(cos(radians(45.0)), cos(radians(30.0)), cosAngle);
}

void main() {
    vec3 rd = normalize(RayDirWorld);
    float denom = dot(rd, plane_normal);
    if (abs(denom) < 0.005) discard;
    float t = (plane_offset - dot(view_pos, plane_normal)) / denom;
    if (t <= 0.0 || !(t < 1e30)) discard;

    vec3 fragPos = view_pos + t * rd;

    vec4 clip = view_proj * vec4(fragPos, 1.0);
    float ndc_depth = (clip.z / clip.w) * 0.5 + 0.5;
    if (ndc_depth >= 0.99999) discard;
    gl_FragDepth = ndc_depth;

    vec3 N = normalize(plane_normal);
    if (dot(N, view_pos - fragPos) < 0.0) N = -N;

    vec3 V = normalize(view_pos - fragPos);
    float NdotV = max(dot(N, V), 0.0);

    vec3 albedo = pow(ground_color, vec3(2.2));
    float roughness = clamp(ground_roughness, 0.0, 1.0);

    // Lambertian direct lighting from the three directional lights — the
    // ground is matte (no specular), so the GGX lobe is omitted. Only the
    // key light (index 0) casts a shadow: the single shadow map represents
    // what the key sees, so gating fill/rim by it too would darken the floor
    // under a shadow those lights never cast (the ghost band the shape
    // shader documents and avoids).
    float shadow0 = ground_shadow(fragPos, N);
    vec3 Lo = vec3(0.0);
    vec3 Lo_unshadowed = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        vec3 L = normalize(light_dirs[i]);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;
        float shadow = (i == 0) ? shadow0 : 0.0;
        vec3 contrib = (albedo / PI) * light_colors[i] * NdotL;
        Lo_unshadowed += contrib;
        Lo += contrib * (1.0 - shadow);
    }

    // Hemispherical ambient, matching the shape shader (matte ground: no specular).
    vec3 up_amb = vec3(0.0, 1.0, 0.0);
    if (up_axis == 0) up_amb = vec3(1.0, 0.0, 0.0);
    if (up_axis == 2) up_amb = vec3(0.0, 0.0, 1.0);
    float sky_fac = dot(N, up_amb) * 0.5 + 0.5;
    vec3 ambient = mix(ambient_ground_color, ambient_sky_color, sky_fac) * albedo * 0.7;

    vec3 color;
    if (shadow_catcher) {
        // Matte shadow catcher (the lookdev presets): there is no visible floor
        // — only the sky, and the shadows cast onto it — matching the USD
        // export's ``primvars:isMatteObject`` floor under RTX.
        //
        // Paint the sky radiance this view ray would have reached, scaled by how
        // far the floor darkens under shadow. ``albedo`` cancels in the ratio
        // (it scales the direct and ambient terms alike), so ``ground_color``
        // cannot tint the result: unshadowed fragments reproduce the sky exactly
        // (ratio 1, floor invisible) and only shadowed ones darken.
        vec3 lit = ambient + Lo_unshadowed;
        vec3 occluded = ambient + Lo;
        vec3 ratio = clamp(occluded / max(lit, vec3(1e-4)), 0.0, 1.0);

        float h_sky = up_axis == 0 ? rd.x : (up_axis == 1 ? rd.y : rd.z);
        float t_sky = clamp(h_sky * 0.5 + 0.5, 0.0, 1.0);
        vec3 sky_lin = mix(pow(max(sky_lower, vec3(0.0)), vec3(2.2)),
                           pow(max(sky_upper, vec3(0.0)), vec3(2.2)), t_sky);
        color = sky_lin * ratio;
    } else {
        // Lookdev-off keeps the pre-lookdev viewer's opaque, spotlit floor.
        color = SpotlightAttenuation(fragPos) * (ambient + Lo);
    }

    // ACES tone mapping (matches the shape shader).
    color = color * exposure;
    vec3 x = color;
    color = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);
    color = pow(color, vec3(1.0 / 2.2));

    // Triangular-PDF dither (~1 LSB) to break up 8-bit banding on smooth ramps.
    color += (fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))
            + fract(52.9829189 * fract(dot(gl_FragCoord.xy + 71.0, vec2(0.06711056, 0.00583715)))) - 1.0) / 255.0;
    FragColor = vec4(color, 1.0);
}
"""


class ShaderEdge(ShaderGL):
    """Flat-color shader for the edge/wireframe overlay pass."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(edge_fragment_shader, "fragment")
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_edge_color = self._get_uniform_location("edge_color")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        edge_color: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0),
        light_space_matrix: np.ndarray | None = None,
    ):
        with self:
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform4f(self.loc_edge_color, *edge_color)
            lsm = light_space_matrix if light_space_matrix is not None else np.eye(4, dtype=np.float32)
            self._gl.glUniformMatrix4fv(self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(lsm))


class ShaderGround(ShaderGL):
    """Shader for the analytical shadow-catcher ground plane."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(ground_vertex_shader, "vertex"),
            Shader(ground_fragment_shader, "fragment"),
        )

        with self:
            self.loc_view_proj = self._get_uniform_location("view_proj")
            self.loc_inv_view_proj = self._get_uniform_location("inv_view_proj")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_view_pos = self._get_uniform_location("view_pos")
            self.loc_up_axis = self._get_uniform_location("up_axis")
            self.loc_plane_normal = self._get_uniform_location("plane_normal")
            self.loc_plane_offset = self._get_uniform_location("plane_offset")
            self.loc_ground_color = self._get_uniform_location("ground_color")
            self.loc_ground_roughness = self._get_uniform_location("ground_roughness")
            self.loc_sun_direction = self._get_uniform_location("sun_direction")
            self.loc_light_dirs = self._get_uniform_location("light_dirs")
            self.loc_light_colors = self._get_uniform_location("light_colors")
            self.loc_shadow_map = self._get_uniform_location("shadow_map")
            self.loc_shadow_radius = self._get_uniform_location("shadow_radius")
            self.loc_shadow_extents = self._get_uniform_location("shadow_extents")
            self.loc_shadow_resolution = self._get_uniform_location("shadow_resolution")
            self.loc_ambient_sky_color = self._get_uniform_location("ambient_sky_color")
            self.loc_ambient_ground_color = self._get_uniform_location("ambient_ground_color")
            self.loc_sky_upper = self._get_uniform_location("sky_upper")
            self.loc_sky_lower = self._get_uniform_location("sky_lower")
            self.loc_exposure = self._get_uniform_location("exposure")
            self.loc_spotlight_enabled = self._get_uniform_location("spotlight_enabled")
            self.loc_shadow_catcher = self._get_uniform_location("shadow_catcher")
