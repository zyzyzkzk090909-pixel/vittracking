from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # ⭐ 只保留一个路径（最关键）
    settings.visdrone2018_path = '/home/pro-c/文档/zy/ORTrack/ORTrack-master/data/visdrone2018'
    settings.prj_dir = '/home/pro-c/文档/zy/ORTrack/ORTrack-master'
    settings.save_dir = '/home/pro-c/文档/zy/ORTrack/ORTrack-master/output'

    return settings
