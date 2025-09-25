from setuptools import find_packages, setup

package_name = 'arob_lab_drones'

setup(
    name=package_name,
    version='0.0.0',
    # package_dir={'': 'src'},
    # packages=find_packages(where='src', exclude=['test']),
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eduardo Montijano',
    maintainer_email='emonti@unizar.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'drone_race = arob_lab_drones.drone_race:main',  
        ],
    },
)
