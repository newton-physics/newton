# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


line_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertexColor;

void main()
{
    vertexColor = aColor;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
"""

line_fragment_shader = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);
}
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
    float worldTexel = 20.0 / float(4096); // world extent / shadow map resolution
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


    float shadow = 0.0;
    float radius = 1.25;
    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    float angle = rand(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rotationMatrix = mat2(c, -s, s, c);
    for(int i = 0; i < 16; i++)
    {
        vec2 offset = rotationMatrix * poissonDisk[i];
        float pcf_depth = texture(shadow_map, projCoords.xy + offset * radius * texelSize).r;
        if(pcf_depth < frag_depth)
            shadow += 1.0;
    }
    shadow /= 16.0;
    return shadow;
}

float SpotlightAttenuation()
{
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

    // surface vectors
    vec3 N = normalize(Normal);
    // Flip normal for backfacing triangles
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 V = normalize(view_pos - FragPos);
    vec3 L = normalize(sun_direction);
    vec3 H = normalize(V + L);

    // Blinn-Phong terms
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    // Derive Blinn-Phong exponent from perceptual roughness.
    // roughness 0 → very rough, 1 → perfectly smooth
    float gloss = clamp(1.0 - roughness, 0.0, 1.0);
    // Map gloss to exponent range ~[2, 1024]
    float shininess = 1.0 + pow(gloss, 4.0) * 1023.0;

    // energy-preserving normalization for Blinn-Phong
    float normFactor = (shininess + 2.0) / (8.0 * PI);

    vec3 diffuse  = albedo * light_color * NdotL * 3.0; // total light intensity multiplier

    // Specular color: dielectrics ~0.04, metals use albedo
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 spec = F0 * light_color * normFactor * pow(NdotH, shininess) * NdotL;

    // simple hemispherical ambient term
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (up_axis == 0) up = vec3(1.0, 0.0, 0.0);
    if (up_axis == 2) up = vec3(0.0, 0.0, 1.0);
    float sky_fac = dot(N, up) * 0.5 + 0.5;
    vec3 ambient = mix(ground_color, sky_color, sky_fac) * albedo;
    // Slight boost for metallics to avoid overly dark mid-roughness metals.
    float metal_ambient_boost = mix(1.0, 1.25, metallic * (1.0 - 0.5 * roughness));
    ambient *= metal_ambient_boost;

    // shadows
    float shadow = ShadowCalculation();

    // spotlight attenuation
    float spotlightAttenuation = SpotlightAttenuation();

    // Metals should contribute little diffuse light.
    diffuse *= 1.0 - metallic;
    vec3 color = ambient + (1.0 - shadow) * spotlightAttenuation * (diffuse + spec);

    // environment reflection for metallic look (fade with roughness)
    float env_lod = clamp(roughness * 4.0, 0.0, 4.0);
    vec3 R = reflect(-V, N);
    vec3 env_color = sample_env_map(R, env_lod);
    env_color = pow(env_color, vec3(2.2)); // to linear
    float reflection_strength = clamp(metallic * pow(1.0 - roughness, 2.0), 0.0, 1.0);
    vec3 env_tint = mix(vec3(1.0), albedo, metallic);
    vec3 env_reflection = env_color * env_tint * env_intensity;
    color = mix(color, env_reflection, reflection_strength);

    // fog
    float dist = length(FragPos - view_pos);
    float fog_start = 20.0;
    float fog_end   = 200.0;
    float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    color = mix(color, pow(fogColor, vec3(2.2)), fog_factor);

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

void main()
{
    float height = max(0.0, (FragPos.z/far_plane));//*0.5 + 0.5);
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

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        view_pos: tuple[float, float, float],
        fog_color: tuple[float, float, float],
        up_axis: int,
        sun_direction: tuple[float, float, float],
        light_color: tuple[float, float, float] = (2.0, 2.0, 2.0),
        ground_color: tuple[float, float, float] = (0.294, 0.333, 0.592),
        sky_color: tuple[float, float, float] = (0.745, 0.863, 0.941),
        enable_shadows: bool = False,
        shadow_texture: int | None = None,
        light_space_matrix: np.ndarray | None = None,
        env_texture: int | None = None,
        env_intensity: float = 1.0,
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
                self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, 0)
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

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_pos: tuple[float, float, float],
        camera_far: float,
        sky_upper: tuple[float, float, float],
        sky_lower: tuple[float, float, float],
        sun_direction: tuple[float, float, float],
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


class ShaderLine(ShaderGL):
    """Simple shader for rendering lines with per-vertex colors."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(line_vertex_shader, "vertex"), Shader(line_fragment_shader, "fragment")
        )

        # Get uniform locations
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")

    def update(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Update view and projection matrices for line rendering."""
        with self:
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
