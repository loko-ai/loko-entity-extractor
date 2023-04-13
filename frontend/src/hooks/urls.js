import axios from "axios";
import urljoin from "url-join";

export class URLObject {
  constructor(url, instance) {
    this.url = url;
    this.instance = instance || axios;
    return new Proxy(this, {
      get: (object, key) => {
        if (key in object) {
          return object[key];
        }
        if (key in this.instance) {
          return function () {
            return this.instance[key](this.url, ...arguments);
          };
        } else {
          return new this.constructor(urljoin(object.url, key), this.instance);
        }
      },
    });
  }
}
export function postFile(client, file, name = "file") {
  const form = new FormData();
  form.append(name, file);
  //state.url = URL.createObjectURL(file);
  return client.post(form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
}
