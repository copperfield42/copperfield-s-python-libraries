"""Scrip para descargar videos de YouTube con youtube_dl.

Notas
>youtube_dl
instalcion: 
    pip install --upgrade pip
    pip install --upgrade youtube_dl

>ffmpeg
instalacion: ir a https://ffmpeg.zeranoe.com/builds/
descargar uno de los build, como el estatico correspondiente,
extraerlo los archivos en la subcarpeta "bin" en una carpeta
conveniente de las se encuentran en la variable de entorno
del sistema PATH, o agregar dicha carpeta a dicha variable.

para comprobar que esto fue exitoso, ejecutar el comando
$>where ffmpeg
debe regresar la donde extragiste estos archivos

29-9-2020:
ir a https://www.ffmpeg.org/ descargar uno de los builds y 
realizar el mismo procedimiento antes mensionado
en particular: https://github.com/BtbN/FFmpeg-Builds/releases
-> ffmpeg-n4.3.1-18-g6d886b6586-win64-gpl

(glp>lglp)

----------------------------------
algunos comandos para youtube-dl

-F                              para ver una lista de los formatos disponibles
-f n                            para descargar el formato espesificado de entre los listados con -F 
   

--skip-download
--write-sub                      Write subtitle file
--write-auto-sub                 Write automatic subtitle file (YouTube only)
--all-subs                       Download all the available subtitles of the video
--list-subs                      List all available subtitles for the video
--sub-format FORMAT              Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best"
--sub-lang LANGS                 Languages of the subtitles to download (optional) separated by commas, use IETF language tags like 'en,pt'
--write-description              Write video description to a .description
                                 file

--write-info-json                Write video metadata to a .info.json file
--write-annotations              Write video annotations to a
                                 .annotations.xml file
--write-thumbnail                Write thumbnail image to disk
--write-all-thumbnails           Write all thumbnail image formats to disk
--list-thumbnails                Simulate and list all available thumbnail
                                 formats           
--download-archive FILE              Download only videos not listed in the
                                     archive file. Record the IDs of all
                                     downloaded videos in it.
                                 
https://github.com/ytdl-org/youtube-dl/blob/master/README.md

miniatura:
https://i.ytimg.com/vi/{id}/maxresdefault.jpg


------------------------------
2020-10-23
En caso que youtube-dl deje de funcionar correctamente por falta de actualizasiones 
debido a que fue tumbado por una DMCA
https://www.reddit.com/r/Python/comments/jgvqa9/the_youtubedl_github_repo_has_received_a_dmca

probar con pytube
https://pypi.org/project/pytube/
https://github.com/nficano/pytube/issues/672
https://pypi.org/project/pytube3/
instalacion:
    pip install --upgrade pip
    pip install --upgrade pytube3
    
https://pypi.org/project/twitter-dl/
pip3 install twitter-dl

2020-10-24
https://youtube-dl-sources.org/
posible alternativa para youtube-dl es youtube-dlc
https://github.com/blackjack4494/yt-dlc
instalcion: 
    pip install --upgrade pip
    pip install --upgrade youtube-dlc


"""

extrar_argv_help="""
Algunos comandos para youtube-dl: -F (para ver una lista de los formatos disponibles)
-f n (para descargar el formato espesificado de entre los listados con -F) 
--skip-download
--write-sub (Write subtitle file)
--write-auto-sub (Write automatic subtitle file (YouTube only))
--all-subs (Download all the available subtitles of the video)
--list-subs (List all available subtitles for the video)
--sub-format FORMAT (Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best")
--sub-lang LANGS (Languages of the subtitles to download (optional) separated by commas, use IETF language tags like 'en,pt')
--write-description (Write video description to a .description file)
--write-info-json (Write video metadata to a .info.json file)
--write-annotations (Write video annotations to a .annotations.xml file)
--write-thumbnail (Write thumbnail image to disk)
--write-all-thumbnails (Write all thumbnail image formats to disk)
--list-thumbnails (Simulate and list all available thumbnail formats)
--download-archive FILE              Download only videos not listed in the
                                     archive file. Record the IDs of all
                                     downloaded videos in it.

"""


import optparse
import youtube_dl
from valid_filenames import valid_file_name
from contextlib_recipes import redirect_folder

__all__ =[ "descargar_video" ]

def descargar_video(url, nombre=None, path=".", extra_arg=(), titulo=False, 
                    id_vid=False, solo_audio=False, write_description=False,
                    skip_download=False, formato="", record_file="",
                    ):
    """Descarga el video espesificado en la url desde twitter, instagam o youtube, con el nombre dado en la carpeta espesificada.
       Si el nombre no es espesificado, se le pondra por nombre su titulo e idintificador.
       extra_arg son adicionales comand line argumentos para la libreria usada para la descarga"""
    comand = [url] if url else []
    if isinstance(url,list):
        comand = url
        nombre=None
    if not nombre:
        comand.extend(["-o","%(title)s [%(id)s].%(ext)s"])
    else:
        comand.extend(["-o",valid_file_name(nombre)+(" - %(title)s" if titulo else "")+(" [%(id)s]" if id_vid else "")+".%(ext)s"])
    if solo_audio:
        comand.extend(["-f",'m4a/bestaudio'])
    else:
        if formato:
            comand.extend(["-f",formato])
    if write_description:
        comand.append("--write-description")
    if skip_download:
        comand.append("--skip-download")
    if record_file:
        comand.extend(["--download-archive",record_file])
    comand.extend(extra_arg)
    print(path)
    print(comand)
    with redirect_folder(path):
        youtube_dl.main(comand)

def descargar_video_filelist(filelist,**kargv):
    """descarga video de ina list en un txt"""
    with open(filelist) as file:
        url_list=list(filter(None, map(str.strip,file)))
        descargar_video(url_list,**kargv)
        
    

def descargar_video_pytube(yt_url,retries=10):
    import pytube
    from pytube import cli
    import tqdm, sys, time
    tqdm_bar = tqdm.tqdm_gui if 'idlelib' in sys.modules else tqdm.tqdm
    print("obteniendo data:" ,repr(yt_url))
    for i in range(1,1+retries):
        try:
            yt = pytube.YouTube(yt_url)
            break
        except KeyError as exc:
            if i==retries:
                print("maximo numero de reintentos alcansado")
                raise
            print(f"error de adquisicion de data en intento {i}/{retries}")
            print(exc)
            time.sleep(1)
    print("data adquirida")
    with tqdm_bar(unit="B", unit_scale=True, leave=False, miniters=1) as progress_bar:
        def on_progress(stream, chunk: bytes, bytes_remaining: int):
            progress_bar.total = stream.filesize
            progress_bar.update(len(chunk))
        yt.register_on_progress_callback(on_progress)
        video = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc()[0]
        return video.download(filename_prefix=f"[{yt.video_id}] ")
    #https://stackoverflow.com/questions/63533960/how-to-cancel-and-pause-a-download-using-pytube
    #https://yagisanatode.com/2018/03/11/how-to-create-a-simple-youtube-download-program-with-a-progress-in-python-3-with-pytube/
    
    
    

def main(argv=None):
    """comand line use"""
    def vararg_callback(option, opt_str, value, parser):
        assert value is None
        value = list(parser.rargs)
        del parser.rargs[:len(value)]
        setattr(parser.values, option.dest, value)
        
    parser    = optparse.OptionParser(usage="%prog url_video(s) [options]")
    parser.add_option("-p","--path")
    parser.add_option("-v","--verbose", action="store_true")
    parser.add_option("-a","--audio", 
                      help="Descarga solo el audio",
                      action="store_true")
    parser.add_option("-n","--name",
                      dest="name",
                      help = "nombre a usar para el/los archivos, de ser omitido se usa el titulo e id del mismo",
                      default =""
                      )                      
    parser.add_option("-t","--title",
                      help="usar titulo del video como nombre de archivo o añadirlo al nombre espesificado",
                      dest="titulo",
                      action="store_true")
    parser.add_option("-i","--id",
                      help="añadir el id del video al nombre espesificado",
                      dest="id_vid",
                      action="store_true")              
    parser.add_option("-w","--write-description",
                      help="escribe en un archivo las description del mismo",
                      dest="writedesc",
                      action="store_true")              
    parser.add_option("-s","--skip-download",
                      help="no descarga el video",
                      dest="skip",
                      action="store_true") 
    parser.add_option("-f","--format",
                      dest="formato",
                      default="mp4",
                      help="formato del video, por defecto mp4, ver documentacion de youtube-dl para detalles, usar '-f \"\"' para usar el valor por defecto de youtube-dl"
                      )
    parser.add_option("--file","--filelist",
                      dest="filelist",
                      help = "archivo de texto con una lista de archivos a descargar",
                      )     
    # parser.add_option("--list",
                      # dest="commandlist",
                      # help="indica que se pasan por linea de comando una list de videos",
                      # action="store_true")
    parser.add_option("--argv","--extra_arg",
                      dest="extra_arg",
                      help = "argumentos para youtube_dl, todo a partir de aqui va para ese modulo."+extrar_argv_help,
                      action = "callback",
                      callback = vararg_callback,
                      default  = () )
    parser.add_option("--record_file",
                      dest="record_file",
                      help = "Registra en este archivo los videos descargados y no descarga los video ya registrados en el mismo"
                       )  
    parser.add_option("--rec",
                      dest="rec",
                      help = "activa la opcion record_file con un archivo llamado 'DESCARGADOS.txt'",
                      action="store_true" )                        
    values, urls = parser.parse_args(argv)
    if len(urls)==1:
        urls=urls[0]
    if values.verbose:
        print(f"{urls=}\nvalues={values}")
    configuracion = dict(path = values.path, 
                         extra_arg = values.extra_arg, 
                         titulo=values.titulo, 
                         solo_audio=values.audio, 
                         id_vid=values.id_vid,
                         write_description=values.writedesc,
                         skip_download=values.skip,
                         formato=values.formato,
                         nombre=values.name,
                         record_file = values.record_file if values.record_file else ("DESCARGADOS.txt" if values.rec else ""),
                        )
    if values.filelist:
        return descargar_video_filelist(values.filelist,**configuracion)
    return descargar_video(urls, **configuracion)
    
if __name__ == "__main__":
    main()
