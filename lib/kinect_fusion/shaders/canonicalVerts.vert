#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexCanonicalPosition;

out vec3 fragCanonicalPosition;

void main()
{

    fragCanonicalPosition = vertexCanonicalPosition;

    vec4 vertexPositionCam = modelViewMatrix*vec4(vertexPosition,1.0);

    gl_Position = projectionMatrix*vertexPositionCam;

}
