from setuptools import find_packages, setup

package_name = 'rokey'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jason',
    maintainer_email='onlysik1290@naver.com',
    description='M0609 협동로봇 웍 조리 실행 노드 (부침개/볶음밥)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 조리 실행 노드. dish 파라미터로 부침개/볶음밥을 고른다.
            #   ros2 run rokey wok_integrate4 --ros-args -p dish:=fried_rice
            #   ros2 run rokey wok_integrate4 --ros-args -p dish:=jeon
            'wok_integrate4 = rokey.wok_integrate4:main',
        ],
    },
)
