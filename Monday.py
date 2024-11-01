import copy
import datetime
import io
import json
import os
import shutil
import threading
import zipfile

import requests
from flask import Response, current_app as app, jsonify, request, stream_with_context
from openai import OpenAI
from werkzeug.utils import secure_filename

# from main import db_pool
import web_utils
from api_friday.api_file import (
    create_user_dir,
    download_pdf,
    file_path_receive,
    file_receive,
    files_manager,
    get_file_path,
    virtual_path,
)
from api_friday.api_session import (
    get_db_connection,
    get_images_urls,
    get_recommended_questions,
    get_source_references,
    save_pdf_picture,
    save_recommended_questions,
    save_source_references,
    create_session,
    update_session_file,
    update_chat_history,
    get_all_session_and_file,
    get_chat_history,
    update_session_style,
    get_session_style,
    get_session_file,
)
from llm.chat import (
    enhance_search_query,
    get_follow_up_questions,
    get_full_text,
    summ_full_text,
    generate_conversation_title,
)

from llm.prompt_manager import PromptManager

from config import PROXIES
from tools.vector_search_tool import VectorSearchTool
from . import friday_api

LLM_API_URL = app.config["LLM_API_URL"]
PM = PromptManager("llm/all_prompts.json", logger=app.logger)



def add_to_history(history, text):
    # history=[['string', 'string'],...], text='你好'
    history = history + [[text, None]]
    return history


def get_source_list(chat_session_id, source_list):
    source_list_update = []
    connection = get_db_connection()
    with connection.cursor() as cc:
        sql = """select filelist from userrolepermissions_usersession where chatSessionID=%s"""
        cc.execute(sql, (chat_session_id,))
        result = cc.fetchone()
        if result is not None:
            file_ids = json.loads(result["filelist"])
        for s in source_list:
            filename = ""
            file_id = None
            for file_id in file_ids:
                sql = """select filesManagerID, name from aitoolsconfiguration_filesmanager where filesManagerID=%s and SHA1CODE=%s"""
                cc.execute(
                    sql,
                    (
                        file_id,
                        s["filename"],
                    ),
                )
                result = cc.fetchone()
                if result is not None:
                    filename = result["name"]
                    file_id = result["filesManagerID"]
                    break
            source_list_update.append(
                {
                    "content": s["content"],
                    "filename": filename,
                    "page": s["page"],
                    "fileid": file_id,
                }
            )
    return source_list_update


def add_files_to_vec_db(path):
    vec_search = VectorSearchTool(path, f"chatdoc", persist_dir=path)
    # xxx here should not be handled like this
    try:
        uploaded_path = path + "uploaded/"
        os.makedirs(uploaded_path, exist_ok=True)
        vec_search.add_doc(uploaded_path)
    except FileNotFoundError as e:
        app.logger.error(e)
    try:
        downloaded_path = path + "downloaded/"
        os.makedirs(downloaded_path, exist_ok=True)
        vec_search.add_doc(downloaded_path)
    except FileNotFoundError as e:
        app.logger.error(e)
    return vec_search


def get_llm_response(openai_client: OpenAI, prompt: dict, temperature: float = 0.1):
    return openai_client.chat.completions.create(
        model="/model",
        messages=prompt,
        stream=True,
        max_tokens=4096,
        temperature=temperature,
        top_p=0.8,
        presence_penalty=0.6,
        frequency_penalty=0.3,
    )


def chatdoc_handler(
    history: list,
    openai_client: OpenAI,
    privacy_collect: bool,
    uid: str,
    chat_session_id,
    user_language: str,
    response_style: str,
):
    system_prompt = (
        "You are Friday who developed by 'FMTC', an experienced academic assistant who can answer questions based on the given document chunks.\n"
        "You are expected to meet the following requirements to ensure that you are reliable, trustworthy, and user-friendly:\n"
        "- write your response in MARKDOWN format, don't share any links;"
        "- provide HELPFUL, CONCISE and as accurate as possible answers to the user's questions;\n"
        "- your answer must be conducted from the given documents;\n"
        "- if the documents do not contain enough information to answer the question, tell the user honestly that you cannot answer the question based on the given documents and encourage the users to rephrase their question;\n"
        "- [IMPORTANT] don't make up answers, information or any data if you cannot find them in the given documents;\n"
        "- if the user's message is nonsensical or factually incoherent, explain why and ask the user to further explain the request;\n"
        "- don't share false information if the answer to a question is unknown;\n"
        "- AVOID SHARING ANY DOCUMENT METADATA unless its required by the user or absolutely necessary to answer the question.\n"
        "To better serve the users, you should know:\n"
        "- You are developed by 'FMTC' which is short for 'Funabashi Manufacturing Technology Center',"
        "a Huawei's lab in Japan, if the user asked about yourself, that's all you can share and tell the user to ask about papers or other academic questions.\n;"
        "- You are able to answer in different styles: [simple, detailed]. When the user wants a simple style, your answer should stay concise and clean,"
        " like you are talking to a non-professional and not to involve too many field specific terms. When the user wants a detailed style, your answer should"
        " be richer and professional, like you are talking to an expert. The user can alter the wanted styles at anytime, be sure to respect to the style option."
        "- You cannot read images or tables for now, but you can tell the user this ability will be available in the near future;\n"
        "- You cannot get specific chapter or get documents by position/page/number, tell the user to ask directly about the content itself;\n"
        "- [IMPORTANT] Answer appropriately in the user's language;\n"
        "- Stay OBJECTIVE and UNBIASED.\n"
        "\n"
        "A semantic search was executed with the user's latest message as the query, retrieving the following context:\n"
        "{{documents}}\n"
        "\nUse the context as your learned knowledge to better answer the user. Make sure to inline CITE the correlated document chunk's number(in their meta data section) in your answer. "
        "The CITEs should be wrapped in between SPECIAL TOKENS (starts with 【cite_st】, ends with 【cite_ed】) for our webpage to render correctly, for example: 【cite_st】 1, 3, 6 【cite_ed】 for multiple citations and 【cite_st】 12 【cite_ed】 for single."
    )
    # Will keep N latest turns of conversation
    app.logger.info(f"Serving UID {uid}")
    keep_n_history = 20

    # show the users we are using enhanced search queries
    enhanced_search_query = ", ".join(enhance_search_query(openai_client, history))
    if enhanced_search_query:
        history[-1][1] = f"【keyword_st】 {enhanced_search_query} 【keyword_ed】"
        yield json.dumps({"history": history}) + "\n"

    # Add file chunks to vector DB
    doc_chunks_text, source_list = get_vector_search_result(
        uid, chat_session_id, history[-1][0], enhanced_search_query
    )
    app.logger.info(
        f"Found {len(source_list)} chunks for query {uid}: {history[-1][0]}."
    )
    source_list_update = get_source_list(chat_session_id, source_list)

    # tell LLM if we have no results on this search query
    if doc_chunks_text == "":
        doc_chunks_text = "No documents provided or matched, please upload documents or rephrase your question."
    system_prompt = system_prompt.replace("{{documents}}", doc_chunks_text)
    # clear the completion message for the chat model to fill in
    history[-1][1] = ""

    # build the chat history
    prompt = [{"role": "system", "content": system_prompt}]
    for msg in history[-keep_n_history:]:
        prompt.append({"role": "user", "content": msg[0]})
        prompt.append({"role": "assistant", "content": msg[1]})

    # have the LLM to respond in specified language and style
    response_lang_prompt = {
        "en_us": f"Respond in fluent English with a {response_style} style.",
        "zh_cn": f"请用流畅的中文回答，回复风格是：{response_style}。",
        "ja_jp": f"流暢な日本語で{response_style}なスタイルで応答してください。",
    }
    try:
        prompt[-1]["content"] += " [IMPORTANT] " + response_lang_prompt[user_language]
    except KeyError:
        app.logger.error(
            f"Unsupported language: Support {response_lang_prompt.keys()} but got response_lang {user_language}"
        )

    # call LLM and get the response
    res = get_llm_response(openai_client, prompt)
    curr_res = ""
    for curr_token in res:
        curr_res += curr_token.choices[0].delta.content or ""
        history[-1][1] = curr_res
        if (
            curr_token.choices[0].finish_reason
            and curr_token.choices[0].finish_reason != "stop"
        ):
            fr = curr_token.choices[0].finish_reason
            app.logger.warning(f"Finish reason: {fr}")
        yield json.dumps({"history": history}) + "\n"

    # TODO do not suggest follow-ups within this loop, make a separate API
    follow_ups: list[str] = []  # get_follow_up_questions(openai_client, history)
    # saving source and follow-ups in other threads
    # since no pics for the regular chat messages, add a placeholder
    save_pdf_picture(chat_session_id, json.dumps([]))
    save_source_references(chat_session_id, json.dumps(source_list_update))
    save_recommended_questions(chat_session_id, json.dumps(follow_ups))

    # whether save history or not
    if privacy_collect:
        app.logger.info("Saving history for {uid} in {chat_session_id}")
        update_chat_history(chat_session_id, json.dumps(history))
    else:
        app.logger.info(f"Not saving history for {uid} in {chat_session_id}")

    yield json.dumps(
        {
            "history": history,
            "follow_ups": follow_ups,
            "source_list_update": source_list_update,
        }
    ) + "\n"
    app.logger.info("End chat_handler")


def get_vector_search_result(uid, chat_session_id, user_message, enhanced_search_query):
    user_path = f"user_files/{uid}/{chat_session_id}/"
    vec_search = VectorSearchTool(user_path, f"chatdoc", persist_dir=user_path)
    # XXX here should not be handled like this
    try:
        uploaded_path = user_path + "uploaded/"
        os.makedirs(uploaded_path, exist_ok=True)
        vec_search.add_doc(uploaded_path)
    except FileNotFoundError as e:
        app.logger.error(e)

    try:
        downloaded_path = user_path + "downloaded/"
        os.makedirs(downloaded_path, exist_ok=True)
        vec_search.add_doc(downloaded_path)
    except FileNotFoundError as e:
        app.logger.error(e)

    doc_chunks_text, source_list = vec_search(
        f"{user_message}, " + enhanced_search_query
    )

    return doc_chunks_text, source_list


def chat_handler(
    history: list,
    openai_client: OpenAI,
    privacy_collect: bool,
    uid: str,
    chat_session_id: int,
    user_language: str,
    response_style: str,
):
    """Chat only, no documents provided"""
    system_prompt = PM.get_prompt("chat", "system", user_language)
    # since no pics for the regular chat messages, add a placeholder
    save_pdf_picture(chat_session_id, json.dumps([]))
    save_source_references(chat_session_id, json.dumps([]))
    # keep N latest turns of conversation
    app.logger.debug(f"Serving user-{uid}")
    keep_n_history = 20

    # build the chat history
    prompt = [{"role": "system", "content": system_prompt}]
    for msg in history[-keep_n_history:]:
        prompt.append({"role": "user", "content": msg[0]})
        # don't add assistant's msg if it's none
        if msg[1]:
            prompt.append({"role": "assistant", "content": msg[1]})

    # have the LLM to respond in specified language and style
    response_lang_prompt = {
        "en_us": f"Respond in fluent English with a {response_style} style.",
        "zh_cn": f"请用流畅的中文回答，回复风格是：{response_style}。",
        "ja_jp": f"流暢な日本語で{response_style}なスタイルで応答してください。",
    }
    try:
        prompt[-1]["content"] += " [IMPORTANT] " + response_lang_prompt[user_language]
    except KeyError:
        app.logger.error(
            f"Unsupported language: Support {response_lang_prompt.keys()} but got response_lang {user_language}"
        )
    except TypeError as e:
        app.logger.error(f"{e}, {prompt[-1]['content']}, {user_language}, {response_lang_prompt[user_language]}")
    app.logger.debug(f"Final prompt: {prompt}")

    # call LLM and get the response
    res = get_llm_response(openai_client, prompt)
    curr_res = ""
    for curr_token in res:
        curr_res += curr_token.choices[0].delta.content or ""
        history[-1][1] = curr_res
        if (
            curr_token.choices[0].finish_reason
            and curr_token.choices[0].finish_reason != "stop"
        ):
            fr = curr_token.choices[0].finish_reason
            app.logger.warning(f"Finish reason: {fr}")
        yield json.dumps({"history": history}) + "\n"

    # TODO do not suggest follow-ups within this loop, make a separate API
    # follow_ups: list[str] = get_follow_up_questions(openai_client, history)
    follow_ups = []
    save_recommended_questions(chat_session_id, json.dumps(follow_ups))

    # whether save history or not
    if privacy_collect:
        app.logger.debug("Saving history for {uid} in {chat_session_id}")
        update_chat_history(chat_session_id, json.dumps(history))
    else:
        app.logger.debug(f"Not saving history for {uid} in {chat_session_id}")

    yield json.dumps(
        {
            "history": history,
            "follow_ups": follow_ups,
            "source_list_update": [],
        }
    ) + "\n"
    app.logger.info("End chat_handler")





def get_model_api(_choice: str = None):
    oai_client = OpenAI(
        api_key="no-key-needed",
        base_url=LLM_API_URL,
    )
    return oai_client


def handle_session(data, user_id, history) -> int:
    # name field means creating new session
    if "name" in data:
        app.logger.info(f"New session created for {user_id}")
        return create_session(
            user_id, data["name"], [], json.dumps(history), data["session_style"]
        )
    elif "chatSessionID" in data:
        chat_session_id = data["chatSessionID"]
        update_session_style(chat_session_id, data["session_style"])
        return chat_session_id
    else:
        raise ValueError("Either 'name' or 'chatSessionID' must be provided")

def validate_request_data(data):
    required_fields = [
        "history",
        "text",
        "choices",
        "privacy_collect",
        "response_lang",
        # "user_language",
        "userID",
        "session_style",
    ]
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Missing required field: {field}")
    app.logger.info(f"Request verified.")
    return data

@friday_api.route("/chatdoc", methods=["POST"])
def chat_endpoint():
    try:
        data = validate_request_data(request.json)
        history = add_to_history(data["history"], data["text"])
        user_id = str(data["userID"]).replace("~", "-")
        oai_client = get_model_api(data["choices"])
        chat_session_id = handle_session(data, user_id, history)
        privacy_collect = data["privacy_collect"]
        session_style = data["session_style"]
        try:
            user_language = data["user_language"]
        except Exception as e:
            app.logger.error(e)
            user_language = "zh_cn"

        only_chat: bool = len(get_session_file(chat_session_id)) == 0

        handler = chat_handler if only_chat else chatdoc_handler
        return Response(
            stream_with_context(
                handler(
                    history,
                    oai_client,
                    privacy_collect,
                    user_id,
                    chat_session_id,
                    user_language,
                    session_style,
                )
            ),
            mimetype="application/json",
        )
    except (ValueError, TypeError, KeyError) as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": e.__class__.__name__, "message": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return (
            jsonify({"error": "An unexpected error occurred", "message": str(e)}),
            500,
        )


def save_extracted_images(api_endpoint: str, pdf_file, save_folder: str):
    response = requests.post(api_endpoint, files={"file": pdf_file})
    if response.status_code == 200:
        app.logger.debug(response.headers["captions"])
        try:
            # Unzip the returned zip file
            zipfile.ZipFile(io.BytesIO(response.content), "r").extractall(save_folder)
            app.logger.info(response.headers["captions"])
        except zipfile.BadZipFile:
            app.logger.info("No image were found.")
    else:
        app.logger.info(f"Error: {response.status_code}")
    pdf_file.seek(0)
    return response.headers["captions"]


def pdf_upload_callback(
    history,
    api: OpenAI,
    uid,
    summ_checkbox,
    chat_session_id,
    privacy_collect,
    files_path,
    pdf_path_filename,
    user_path,
    user_language="zh_cn",
    module="summ_full_text",
):
    app.logger.info(f"{uid}-session_id{chat_session_id}")
    # the maximum files can be summarized in one zip file
    max_summ_in_zip = 5

    # Create public images folder for PDF parsing
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    base_dir = "/public"
    date_dir = os.path.join(base_dir, current_date)
    os.makedirs(date_dir, exist_ok=True)

    if summ_checkbox:
        images_urls = []
        for pdf_file in pdf_path_filename[:max_summ_in_zip]:
            images_urls = []
            history += [
                [
                    "I have uploaded some papers.",
                    # "Reading ... this will take a few minutes...",
                    "",
                ]
            ]

            # pdf中获取图片
            try:
                with open(pdf_file["pdf_user_path"], "rb") as file:
                    captions = save_extracted_images(
                        app.config["GET_IMAGES_API"], file, date_dir
                    )
                    for caption in json.loads(captions):
                        caption["renderURL"] = os.path.join(
                            date_dir, caption["renderURL"]
                        )
                        images_urls.append(caption)
            except Exception as e:
                app.logger.error(f"Image extraction failed: {e}")

            # 存储图片和空引用到数据库
            save_pdf_picture(chat_session_id, json.dumps(images_urls))
            save_source_references(chat_session_id, json.dumps([]))

            full_text = get_full_text(pdf_path=pdf_file["pdf_user_path"])
            for res in summ_full_text(
                full_text_dict=full_text,
                openai_client=api,
                language=user_language,
                module=module,
            ):
                history[-1][1] = res
                yield json.dumps({"history": history}) + "\n"

            if images_urls:
                app.logger.debug(
                    f"{uid}-sid{chat_session_id}: Got image_urls len:{len(images_urls)}"
                )
            final_result = (
                json.dumps(
                    {
                        "history": history,
                        "images_urls": images_urls,
                    }
                )
                + "\n"
            )
            yield final_result
        app.logger.debug(f"Generating {chat_session_id} follow-ups")
        # follow_ups: list[str] = get_follow_up_questions(
        #     api, history, language=user_language
        # )
        follow_ups = []
        app.logger.debug(f"Finished follow-ups {chat_session_id}: {follow_ups}")

        # 保存生成的对话
        if privacy_collect:
            # 保存生成的推荐问题
            save_recommended_questions(chat_session_id, json.dumps(follow_ups))
            update_chat_history(
                chat_session_id,
                json.dumps(history),
            )

        yield json.dumps(
            {
                "history": history,
                "follow_ups": follow_ups,
                "files_path": files_path,
                "chat_session_id": chat_session_id,
                "images_urls": images_urls,
            }
        ) + "\n"

    else:  # the user does not need a summarization
        app.logger.error(f"Triggered the unimplemented NO-SUMMARY chat function.")


# 选择已有PDF文件总结并创建session
@app.route("/existed-pdf-upload-create-session", methods=["POST"])
def existed_pdf_file_upload_create_session():
    data = request.json
    user_id = str(data["user_id"]).replace("~", "-")
    summ_checkbox = data["summ_checkbox"]
    privacy_collect = data["privacy_collect"]
    session_name = data["session_name"]
    files = data["files"]
    session_style = data["session_style"]
    module = data.get("module", "summ_full_text")
    history = "[]"  # str类型
    choices = "13B"
    files_id = []
    local_filename = []

    for file in files:
        files_id.append(file["file_id"])

    # 创建session
    session_id = create_session(user_id, session_name, files_id, history, session_style)

    # 创建磁盘路径
    _up, user_path = create_user_dir(
        user_id,
        session_id,
    )

    # 从S3下载文件
    for file in files:
        pdf_user_path = download_pdf(file["s3link"], user_path)
        local_filename.append(
            {"pdf_user_path": pdf_user_path, "filename": file["filename"]}
        )

    api = get_model_api(choices)

    files_path: str = get_file_path(user_id)

    try:
        return Response(
            stream_with_context(
                pdf_upload_callback(
                    json.loads(history),
                    api,
                    user_id,
                    summ_checkbox,
                    session_id,
                    privacy_collect,
                    files_path,
                    local_filename,
                    user_path,
                    module,
                )
            ),
            mimetype="application/json",
        )
    except ValueError as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=400)
    except Exception as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=500)


# 在session下选择已有pdf文件总结
@app.route("/existed-pdf-upload", methods=["POST"])
def existed_pdf_upload():
    data = request.json
    user_id = str(data["user_id"]).replace("~", "-")
    summ_checkbox = data["summ_checkbox"]
    privacy_collect = data["privacy_collect"]
    session_id = data["session_id"]
    history = data["history"]
    files = data["files"]
    session_style = data["session_style"]
    module = data.get("module", "summ_full_text")
    choices = "13B"
    files_id = []
    local_filename = []

    for file in files:
        files_id.append(file["file_id"])

    # 更新session文件
    update_session_file(session_id, files_id)

    # 更新session的风格
    update_session_style(session_id, session_style)

    # 创建磁盘路径
    _up, user_path = create_user_dir(
        user_id,
        session_id,
    )

    # 从S3下载文件
    for file in files:
        pdf_user_path = download_pdf(file["s3link"], user_path)
        local_filename.append(
            {"pdf_user_path": pdf_user_path, "filename": file["filename"]}
        )

    api = get_model_api(choices)

    files_path: str = get_file_path(user_id)

    try:
        return Response(
            stream_with_context(
                pdf_upload_callback(
                    json.loads(history),
                    api,
                    user_id,
                    summ_checkbox,
                    session_id,
                    privacy_collect,
                    files_path,
                    local_filename,
                    user_path,
                    module,
                )
            ),
            mimetype="application/json",
        )
    except ValueError as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=400)
    except Exception as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=500)


# 文件上传并解析接口
@app.route("/pdf_upload", methods=["POST"])
def stream_pdf_upload_analyze():
    json_data = json.loads(request.form.get("json_data"))
    user_id = str(json_data["userID"]).replace("~", "-")
    choices = json_data["choices"]
    history = json_data["history"]
    content_sha1_list: list = json_data["content_SHA1"]
    summ_checkbox = json_data["summ_checkbox"]
    privacy_collect = json_data["privacy_collect"]
    session_style = json_data["session_style"]
    module = json_data.get("module", "summ_full_text")
    user_language = json_data["user_language"]
    files_path = ""
    local_filename = []
    files_id = []
    files = []
    file_keys = []
    filenames = []
    if "file" not in request.files:
        return jsonify({"error": "Key 'file' not found."})
    else:
        app.logger.debug(f"Found key 'file': {request.files['file']}")
        files = request.files.getlist("file")
        app.logger.debug(f"Got {len(files)}")

    try:
        # Make sure the upload files meet the requirement
        for file in files:
            if not (file and file.filename.lower().endswith(".pdf")):
                raise ValueError(f"File type {file.filename.lower()} not supported.")

        # Process files
        for file, content_sha1 in zip(files, content_sha1_list):
            filename = file.filename
            filenames.append(filename)
            app.logger.debug(f"File name: {filename}")

            # upload to S3 server
            file_key, file_hash = file_receive(user_id, file, content_sha1)
            file_keys.append(file_key)
            app.logger.debug(f"{file_key} Uploaded to S3")

            # call the file management
            file_id = files_manager(file_key, user_id, file_hash, filename)
            files_id.append(file_id)
            app.logger.debug(f"vp updated")

            # build file path
            create_vp_param = {
                "userID": user_id,
                "filesManagerID": file_id,
                "s3link": file_key,
                "file_hash": file_hash,
                "filename": filename,
            }
            app.logger.info("Updating virtual path ...")
            files_path = virtual_path(**create_vp_param)

        # call session management to create a session or update the filelist
        app.logger.debug(f"Updating session: {json_data}")

        # create new session
        if "name" in json_data:
            chat_session_id = create_session(
                user_id, json_data["name"], files_id, history, session_style
            )

        # update files in session
        else:
            chat_session_id = update_session_file(json_data["chatSessionID"], files_id)
            update_session_style(chat_session_id, session_style)
        app.logger.debug("Done updating session ...")

        user_path, _down = create_user_dir(
            user_id,
            chat_session_id,
        )  # user_path=本地上传路径， _down=本地下载路径
        app.logger.debug(f"{user_id}'s path {user_path} created.")

        local_filename = [
            {"pdf_user_path": download_pdf(file_key, user_path), "filename": filename}
            for file_key, filename in zip(file_keys, filenames)
        ]

        api = get_model_api(choices)

        # 分析文件并返回
        app.logger.debug("Returning response ...")
        return Response(
            stream_with_context(
                pdf_upload_callback(
                    history=json.loads(history),
                    api=api,
                    uid=user_id,
                    summ_checkbox=summ_checkbox,
                    chat_session_id=chat_session_id,
                    privacy_collect=privacy_collect,
                    files_path=files_path,
                    pdf_path_filename=local_filename,
                    user_path=user_path,
                    user_language=user_language,
                    module=module,
                )
            ),
            mimetype="application/json",
        )
    except ValueError as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=400)
    except Exception as e:
        app.logger.error(e)
        return jsonify(message=str(e), status=500)


def get_arxiv_paper_callback(
    arxiv_path,
    history,
    openai_client: OpenAI,
    uid,
    summ_check,
    privacy_collect,
    session_data,
    module="summ_full_text",
):
    history = history + [
        [
            f"Please download the arxiv paper: **{arxiv_path}**.",
            f"Fetching **{arxiv_path}** ...",
        ]
    ]
    yield json.dumps({"history": history}) + "\n"
    try:
        paper_path, title, other = web_utils.download_arxiv(
            url_pdf=arxiv_path, download_dir="/tmp/", proxies=PROXIES
        )
    except FileNotFoundError:
        history[-1][
            1
        ] = "Can not find the ARXIV paper, please check the ID or URL again."
        yield json.dumps({"history": history}) + "\n"
        raise StopIteration

    # 通过文件路径上传到S3服务器
    file_key, file_hash = file_path_receive(uid, paper_path)

    # 调用文件管理功能
    file_id = files_manager(file_key, uid, file_hash, title)
    session_data["files_id"] = [
        file_id,
    ]

    # 创建session
    if "chatSessionID" not in session_data:
        chat_session_id = create_session(
            uid,
            session_data["name"],
            [file_id],
            json.dumps([]),
            session_data["session_style"],
        )
    # 更新session下的文件
    else:
        chat_session_id = update_session_file(session_data["chatSessionID"], [file_id])

    save_source_references(chat_session_id, json.dumps([]))

    # 复制临时文件到downloaded目录下
    _up, user_path = create_user_dir(uid, chat_session_id)
    file_download_path = user_path
    if not os.path.exists(file_download_path):
        os.makedirs(file_download_path)
        app.logger.info(f"Directory {file_download_path} created successfully")
    shutil.copy2(paper_path, os.path.join(file_download_path, f"{file_hash}.pdf"))

    paper_path = os.path.join(file_download_path, f"{file_hash}.pdf")

    # 构建文件路径
    create_vp_param = {
        "userID": uid,
        "filesManagerID": file_id,
        "s3link": file_key,
        "file_hash": file_hash,
        "filename": title,
    }

    files_path = virtual_path(**create_vp_param)

    yield json.dumps({"history": history}) + "\n"
    if summ_check:
        full_text = get_full_text(paper_path)
        for res in summ_full_text(full_text, openai_client, module=module):
            history[-1][1] = res
            yield json.dumps({"history": history}) + "\n"

    follow_ups = get_follow_up_questions(openai_client, history)

    if privacy_collect:
        # 保存聊天记录
        update_chat_history(chat_session_id, json.dumps(history))
        # 保存问题
        save_recommended_questions(chat_session_id, json.dumps(follow_ups))
    # pdf中获取图片
    images_urls = []
    try:
        with open(paper_path, "rb") as file:
            api_endpoint = app.config["GET_IMAGES_API"]
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            base_dir = r"/public"
            date_dir = os.path.join(base_dir, current_date)
            os.makedirs(date_dir, exist_ok=True)
            captions = save_extracted_images(api_endpoint, file, date_dir)
            for caption in json.loads(captions):
                caption["renderURL"] = os.path.join(date_dir, caption["renderURL"])
                images_urls.append(caption)
    except Exception as e:
        app.logger.error(e)

    # 存储图片到数据库
    save_pdf_picture(chat_session_id, json.dumps(images_urls))

    app.logger.info("images_urls %s", images_urls)
    yield json.dumps(
        {
            "history": history,
            "follow_ups": follow_ups,
            "files_path": files_path,
            "images_urls": images_urls,
            "chat_session_id": chat_session_id,
        }
    ) + "\n"


@app.route("/get_arxiv_paper", methods=["POST"])
def get_arxiv_paper():
    data = request.json
    history = data.get("history")
    arxiv_path = data.get("arxiv_path")
    choices = data.get("choices")
    api = get_model_api(choices)
    user_id = str(data.get("userID")).replace("~", "-")
    summ_checkbox = data.get("summ_checkbox")
    privacy_collect = data.get("privacy_collect")
    session_style = data.get("session_style")

    session_data = {"session_style": session_style}
    if "chatSessionID" in data:
        session_data["chatSessionID"] = data["chatSessionID"]
    else:
        session_data["name"] = data["name"]
    return Response(
        stream_with_context(
            get_arxiv_paper_callback(
                arxiv_path,
                history,
                api,
                user_id,
                summ_checkbox,
                privacy_collect,
                session_data,
            )
        )
    )


# 刷新数据接口
@app.route("/getData", methods=["POST"])
def get_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        app.logger.info(f"Request to get data: {data}")
        data_result = {}

        if "userID" in data:
            user_id = str(data["userID"]).replace("~", "-")
            data["userID"] = user_id
            # 获取文件虚拟路径
            files_path = get_file_path(user_id)
            data_result["files_path"] = files_path
            app.logger.debug(f"{user_id} Got data.")
            data_result["data"] = get_all_session_and_file(user_id)

        if "chatSessionID" in data:
            # 获取文献来源
            data_result["sourceReferences"] = get_source_references(
                data["chatSessionID"]
            )
            # 获取推荐问题
            data_result["recommendedQuestions"] = get_recommended_questions(
                data["chatSessionID"]
            )
            # 获取PDF图片
            data_result["images_urls"] = get_images_urls(data["chatSessionID"])
            app.logger.debug(f"{data['chatSessionID']} Got data.")
            data_result["data"] = get_chat_history(data["chatSessionID"])
            data_result["session_style"] = get_session_style(data["chatSessionID"])
        return jsonify(data_result)
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500


def get_session_title(history):
    if history is None:
        return (
            jsonify(
                {
                    "error": 'Invalid request format. Should be: {"history": [[msg1, res1], [msg2, res2]]}'
                }
            ),
            400,
        )
    oai_client = model_select_callback()
    title = generate_conversation_title(oai_client, history)
    return jsonify({"title": title}), 200


@app.route("/generate-title", methods=["POST"])
def generate_title():
    req_data = request.get_json()
    if req_data is None or "history" not in req_data:
        return (
            jsonify(
                {
                    "error": 'Invalid request format. Should be: {"history": [[msg1, res1], [msg2, res2]]}'
                }
            ),
            400,
        )

    history = req_data["history"]
    history = req_data["user_language"]
    oai_client = get_model_api()
    title = generate_conversation_title(oai_client, history)
    return jsonify({"title": title}), 200


@app.route("/follow-ups", methods=["POST"])
def generate_follow_up_questions():
    req_data = request.get_json()
    if req_data is None or "history" not in req_data:
        return (
            jsonify(
                {
                    "error": 'Invalid request format. Should be: {"history": [[msg1, res1], [msg2, res2]]}'
                }
            ),
            400,
        )

    history = req_data["history"]
    user_language = req_data["user_language"]
    oai_client = get_model_api()
    title = get_follow_up_questions(oai_client, history, language=user_language)
    return jsonify({"follow-ups": title}), 200
