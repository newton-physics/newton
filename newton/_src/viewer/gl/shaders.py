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
    Normal = mat3(transpose(inverse(rotation))) * aNormal;
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
uniform vec3 light_color;
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
    float worldTexel = (shadow_extents * 2.0) / float(4096); // world extent / shadow map resolution
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
    vec3 L = normalize(sun_direction);
    vec3 H = normalize(V + L);

    // Cook-Torrance PBR
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float HdotV = max(dot(H, V), 0.0);

    // GGX/Trowbridge-Reitz normal distribution
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom);

    // Schlick-GGX geometry function (Smith method for both view and light)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    float G1_L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1_V * G1_L;

    // Schlick Fresnel, dampened by roughness to reduce edge aliasing
    vec3 F_max = mix(F0, vec3(1.0), 1.0 - roughness);
    vec3 F = F0 + (F_max - F0) * pow(1.0 - HdotV, 5.0);

    // Cook-Torrance specular BRDF
    vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    // Diffuse uses remaining energy not reflected
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // Direct lighting
    vec3 Lo = (diffuse * diffuse_scale + spec * specular_scale) * light_color * NdotL * 3.0;

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

    // shadows
    float shadow = ShadowCalculation();

    float spotAttenuation = SpotlightAttenuation();
    vec3 color = ambient + (1.0 - shadow) * spotAttenuation * Lo;

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

uniform vec3 sky_upper;
uniform vec3 sky_lower;
uniform float far_plane;

uniform vec3 sun_direction;
uniform int up_axis;

void main()
{
    float h = up_axis == 0 ? FragPos.x : (up_axis == 1 ? FragPos.y : FragPos.z);
    float height = max(0.0, h / far_plane);
    vec3 sky = mix(sky_lower, sky_upper, height);

    float diff = max(dot(sun_direction, normalize(FragPos)), 0.0);
    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;

    FragColor = vec4(sky + sun, 1.0);
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
            self.loc_light_color = self._get_uniform_location("light_color")
            self.loc_ground_color = self._get_uniform_location("ground_color")
            self.loc_sky_color = self._get_uniform_location("sky_color")
            self.loc_shadow_radius = self._get_uniform_location("shadow_radius")
            self.loc_diffuse_scale = self._get_uniform_location("diffuse_scale")
            self.loc_specular_scale = self._get_uniform_location("specular_scale")
            self.loc_spotlight_enabled = self._get_uniform_location("spotlight_enabled")
            self.loc_shadow_extents = self._get_uniform_location("shadow_extents")
            self.loc_exposure = self._get_uniform_location("exposure")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        view_pos: tuple[float, float, float],
        fog_color: tuple[float, float, float],
        up_axis: int,
        sun_direction: tuple[float, float, float],
        light_color: tuple[float, float, float] = (2.0, 2.0, 2.0),
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
            self._gl.glUniform3f(self.loc_light_color, *light_color)
            self._gl.glUniform3f(self.loc_ground_color, *ground_color)
            self._gl.glUniform3f(self.loc_sky_color, *sky_color)
            self._gl.glUniform1f(self.loc_shadow_radius, shadow_radius)
            self._gl.glUniform1f(self.loc_diffuse_scale, diffuse_scale)
            self._gl.glUniform1f(self.loc_specular_scale, specular_scale)
            self._gl.glUniform1i(self.loc_spotlight_enabled, int(spotlight_enabled))
            self._gl.glUniform1f(self.loc_shadow_extents, shadow_extents)
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
            self.loc_sun_direction = self._get_uniform_location("sun_direction")
            self.loc_up_axis = self._get_uniform_location("up_axis")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_pos: tuple[float, float, float],
        camera_far: float,
        sky_upper: tuple[float, float, float],
        sky_lower: tuple[float, float, float],
        sun_direction: tuple[float, float, float],
        up_axis: int = 2,
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
            self._gl.glUniform3f(self.loc_sun_direction, *sun_direction)
            self._gl.glUniform1i(self.loc_up_axis, up_axis)


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
