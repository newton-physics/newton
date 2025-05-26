cmake_minimum_required(VERSION 2.8.3)
project(h1_description)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir launch meshes urdf)
  install(DIRECTORY ${dir}/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
